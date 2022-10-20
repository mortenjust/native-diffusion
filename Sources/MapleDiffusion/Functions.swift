import MetalPerformanceShadersGraph
import Foundation

// Maple Diffusion implements stable diffusion (original v1.4 model)
// inference via MPSGraph. iOS has a hard memory limit of 4GB (with
// a special entitlement), so this implementation trades off latency
// for memory usage in many places (tagged with MEM-HACK) in order to
// stay under the limit and minimize probability of oom.

var modelFolder : URL! // TODO: Make optional, add safety

func makeGraph() -> MPSGraph {
    let graph = MPSGraph()
    graph.options = MPSGraphOptions.none
    return graph
}

func loadConstant(graph: MPSGraph, name: String, shape: [NSNumber], fp32: Bool = false) -> MPSGraphTensor {
    
    let numels = shape.map({$0.intValue}).reduce(1, *)
    
    let fileUrl: URL = modelFolder.appendingPathComponent(name + (fp32 ? "_fp32" : "")).appendingPathExtension("bin")
//    let fileUrl: URL = Bundle.main.url(forResource: "bins/" + name + (fp32 ? "_fp32" : ""), withExtension: ".bin")!
    
    let data: Data = try! Data(contentsOf: fileUrl, options: Data.ReadingOptions.alwaysMapped)
    let expectedCount = numels * (fp32 ? 4 : 2)
    assert(data.count == expectedCount, "Mismatch between byte count of data \(data.count) and expected size \(expectedCount) for \(numels) els in \(fileUrl)")
    return graph.constant(data, shape: shape, dataType: fp32 ? MPSDataType.float32 : MPSDataType.float16)
}

func makeConv(graph: MPSGraph, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, khw: NSNumber, stride: Int = 1, bias: Bool = true) -> MPSGraphTensor {
    let w = loadConstant(graph: graph, name: name + ".weight", shape: [outChannels, xIn.shape![3], khw, khw])
    let p: Int = khw.intValue / 2;
    let convDesc = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: p, paddingRight: p, paddingTop: p, paddingBottom: p, paddingStyle: MPSGraphPaddingStyle.explicit, dataLayout: MPSGraphTensorNamedDataLayout.NHWC, weightsLayout: MPSGraphTensorNamedDataLayout.OIHW)!
    let conv = graph.convolution2D(xIn, weights: w, descriptor: convDesc, name: nil)
    if (bias) {
        let b = loadConstant(graph: graph, name: name + ".bias", shape: [1, 1, 1, outChannels])
        return graph.addition(conv, b, name: nil)
    }
    return conv
}

func makeUpsampleNearest(graph: MPSGraph, xIn: MPSGraphTensor, scaleFactor: Int=2) -> MPSGraphTensor {
    return graph.resize(xIn, size: [NSNumber(value:xIn.shape![1].intValue * scaleFactor), NSNumber(value:xIn.shape![2].intValue * scaleFactor)], mode: MPSGraphResizeMode.nearest, centerResult: true, alignCorners: false, layout: MPSGraphTensorNamedDataLayout.NHWC, name: nil)
}

@available(macOS 12.3, *)
func makeGroupNorm(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = xIn
    if (xIn.shape!.count == 3) {
        x = graph.expandDims(x, axes: [1], name: nil)
    }
    let shape = x.shape!
    let nGroups: NSNumber = 32
    let nGrouped: NSNumber = shape[3].floatValue / nGroups.floatValue as NSNumber
    let gamma = loadConstant(graph: graph, name: name + ".weight", shape: [1, 1, 1, nGroups, nGrouped])
    let beta = loadConstant(graph: graph, name: name + ".bias", shape: [1, 1, 1, nGroups, nGrouped])
    x = graph.reshape(x, shape: [shape[0], shape[1], shape[2], nGroups, nGrouped], name: nil)
    let mean = graph.mean(of: x, axes: [1, 2, 4], name: nil)
    let variance = graph.variance(of: x, axes: [1, 2, 4], name: nil)
    x = graph.normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
    return graph.reshape(x, shape: xIn.shape!, name: nil)
}

func makeSwish(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    return graph.multiplication(xIn, graph.sigmoid(with: xIn, name: nil), name: nil)
}

@available(macOS 12.3, *)
func makeGroupNormSwish(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    return makeSwish(graph: graph, xIn: makeGroupNorm(graph: graph, xIn: xIn, name: name))
}

@available(macOS 12.3, *)
func makeDecoderResBlock(graph: MPSGraph, xIn: MPSGraphTensor, name: String, outChannels: NSNumber) -> MPSGraphTensor {
    var x = xIn
    x = makeGroupNormSwish(graph: graph, xIn: x, name: name + ".norm1")
    x = makeConv(graph: graph, xIn: x, name: name + ".conv1", outChannels: outChannels, khw: 3)
    x = makeGroupNormSwish(graph: graph, xIn: x, name: name + ".norm2")
    x = makeConv(graph: graph, xIn: x, name: name + ".conv2", outChannels: outChannels, khw: 3)
    if (xIn.shape![3] != outChannels) {
        let ninShortcut = makeConv(graph: graph, xIn: xIn, name: name + ".nin_shortcut", outChannels: outChannels, khw: 1)
        return graph.addition(x, ninShortcut, name: "skip")
    }
    return graph.addition(x, xIn, name: "skip")
}

@available(macOS 12.3, *)
func makeDecoderAttention(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = makeGroupNorm(graph: graph, xIn: xIn, name: name + ".norm")
    let c = x.shape![3]
    x = graph.reshape(x, shape: [x.shape![0], NSNumber(value:x.shape![1].intValue * x.shape![2].intValue), c], name: nil)
    let q = makeLinear(graph: graph, xIn: x, name: name + ".q", outChannels: c, bias: false)
    var k = makeLinear(graph: graph, xIn: x, name: name + ".k", outChannels: c, bias: false)
    k = graph.multiplication(k, graph.constant(1.0 / sqrt(c.doubleValue), dataType: MPSDataType.float16), name: nil)
    k = graph.transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
    let v = makeLinear(graph: graph, xIn: x, name: name + ".v", outChannels: c, bias: false)
    var att = graph.matrixMultiplication(primary: q, secondary: k, name: nil)
    att = graph.softMax(with: att, axis: 2, name: nil)
    att = graph.matrixMultiplication(primary: att, secondary: v, name: nil)
    x = makeLinear(graph: graph, xIn: att, name: name + ".proj_out", outChannels: c)
    x = graph.reshape(x, shape: xIn.shape!, name: nil)
    return graph.addition(x, xIn, name: nil)
}

func makeByteConverter(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    var x = xIn
    x = graph.clamp(x, min: graph.constant(0, shape: [1], dataType: MPSDataType.float16), max: graph.constant(1.0, shape: [1], dataType: MPSDataType.float16), name: nil)
    x = graph.multiplication(x, graph.constant(255, shape: [1], dataType: MPSDataType.float16), name: nil)
    x = graph.round(with: x, name: nil)
    x = graph.cast(x, to: MPSDataType.uInt8, name: "cast to uint8 rgba")
    let alpha = graph.constant(255, shape: [1,  x.shape![1], x.shape![2], 1], dataType: MPSDataType.uInt8)
    return graph.concatTensors([x, alpha], dimension: 3, name: nil)
}

@available(macOS 12.3, *)
func makeDecoder(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    var x = xIn
    let name = "first_stage_model.decoder"
    x = graph.multiplication(x, graph.constant(1 / 0.18215, dataType: MPSDataType.float16), name: "rescale")
    x = makeConv(graph: graph, xIn: x, name: "first_stage_model.post_quant_conv", outChannels: 4, khw: 1)
    x = makeConv(graph: graph, xIn: x, name: name + ".conv_in", outChannels: 512, khw: 3)
    
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".mid.block_1", outChannels: 512)
    x = makeDecoderAttention(graph: graph, xIn: x, name: name + ".mid.attn_1")
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".mid.block_2", outChannels: 512)
    
    // block 3
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.3.block.0", outChannels: 512)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.3.block.1", outChannels: 512)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.3.block.2", outChannels: 512)
    x = makeUpsampleNearest(graph: graph, xIn: x)
    x = makeConv(graph: graph, xIn: x, name: name + ".up.3.upsample.conv", outChannels: 512, khw: 3)
    
    // block 2
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.2.block.0", outChannels: 512)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.2.block.1", outChannels: 512)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.2.block.2", outChannels: 512)
    x = makeUpsampleNearest(graph: graph, xIn: x)
    x = makeConv(graph: graph, xIn: x, name: name + ".up.2.upsample.conv", outChannels: 512, khw: 3)
    
    // block 1
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.1.block.0", outChannels: 256)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.1.block.1", outChannels: 256)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.1.block.2", outChannels: 256)
    x = makeUpsampleNearest(graph: graph, xIn: x)
    x = makeConv(graph: graph, xIn: x, name: name + ".up.1.upsample.conv", outChannels: 256, khw: 3)
    // block 0
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.0.block.0", outChannels: 128)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.0.block.1", outChannels: 128)
    x = makeDecoderResBlock(graph: graph, xIn: x, name: name + ".up.0.block.2", outChannels: 128)
    
    x = makeGroupNormSwish(graph: graph, xIn: x, name: name + ".norm_out")
    x = makeConv(graph: graph, xIn: x, name: name + ".conv_out", outChannels: 3, khw: 3)
    x = graph.addition(x, graph.constant(1.0, dataType: MPSDataType.float16), name: nil)
    x = graph.multiplication(x, graph.constant(0.5, dataType: MPSDataType.float16), name: nil)
    return makeByteConverter(graph: graph, xIn: x)
}

func makeLayerNorm(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    assert(xIn.shape!.count == 3, "layernorm requires NTC")
    let gamma = loadConstant(graph: graph, name: name + ".weight", shape: [1, 1, xIn.shape![2]])
    let beta = loadConstant(graph: graph, name: name + ".bias", shape: [1,  1, xIn.shape![2]])
    let mean = graph.mean(of: xIn, axes: [2], name: nil)
    let variance = graph.variance(of: xIn, axes: [2], name: nil)
    let x = graph.normalize(xIn, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
    return graph.reshape(x, shape: xIn.shape!, name: nil)
}

func makeLinear(graph: MPSGraph, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, bias: Bool = true) -> MPSGraphTensor {
    if (xIn.shape!.count == 2) {
        var x = graph.reshape(xIn, shape: [xIn.shape![0], 1, 1, xIn.shape![1]], name: nil)
        x = makeConv(graph: graph, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
        return graph.reshape(x, shape: [xIn.shape![0], outChannels], name: nil)
    }
    var x = graph.reshape(xIn, shape: [xIn.shape![0], 1, xIn.shape![1], xIn.shape![2]], name: nil)
    x = makeConv(graph: graph, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
    return graph.reshape(x, shape: [xIn.shape![0], xIn.shape![1], outChannels], name: nil)
}

func makeTimeEmbed(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = xIn
    x = makeLinear(graph: graph, xIn: x, name: name + ".0", outChannels: 1280)
    x = makeSwish(graph: graph, xIn: x)
    return makeLinear(graph: graph, xIn: x, name: name + ".2", outChannels: 1280)
}

@available(macOS 12.3, *)
func makeUNetResBlock(graph: MPSGraph, xIn: MPSGraphTensor, embIn: MPSGraphTensor, name: String, inChannels: NSNumber, outChannels: NSNumber) -> MPSGraphTensor {
    var x = xIn
    x = makeGroupNormSwish(graph: graph, xIn: x, name: name + ".in_layers.0")
    x = makeConv(graph: graph, xIn: x, name: name + ".in_layers.2", outChannels: outChannels, khw: 3)
    var emb = embIn
    emb = makeSwish(graph: graph, xIn: emb)
    emb = makeLinear(graph: graph, xIn: emb, name: name + ".emb_layers.1", outChannels: outChannels)
    emb = graph.expandDims(emb, axes: [1, 2], name: nil)
    x = graph.addition(x, emb, name: nil)
    x = makeGroupNormSwish(graph: graph, xIn: x, name: name + ".out_layers.0")
    x = makeConv(graph: graph, xIn: x, name: name + ".out_layers.3", outChannels: outChannels, khw: 3)
    
    var skip = xIn
    if (inChannels != outChannels) {
        skip = makeConv(graph: graph, xIn: xIn, name: name + ".skip_connection", outChannels: outChannels, khw: 1)
    }
    return graph.addition(x, skip, name: nil)
}

func makeCrossAttention(graph: MPSGraph, xIn: MPSGraphTensor, name: String, context: MPSGraphTensor?, saveMemory: Bool) -> MPSGraphTensor {
    let c = xIn.shape![2]
    let (nHeads, dHead) = (NSNumber(8), NSNumber(value: c.intValue / 8))
    var q = makeLinear(graph: graph, xIn: xIn, name: name + ".to_q", outChannels: c, bias: false)
    let context = context ?? xIn
    var k = makeLinear(graph: graph, xIn: context, name: name + ".to_k", outChannels: c, bias: false)
    var v = makeLinear(graph: graph, xIn: context, name: name + ".to_v", outChannels: c, bias: false)
    let n = xIn.shape![0]
    let hw = xIn.shape![1]
    let t = context.shape![1]
    q = graph.reshape(q, shape: [n, hw, nHeads, dHead], name: nil)
    k = graph.reshape(k, shape: [n, t, nHeads, dHead], name: nil)
    v = graph.reshape(v, shape: [n, t, nHeads, dHead], name: nil)
    
    q = graph.transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
    k = graph.transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
    k = graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil)
    k = graph.multiplication(k, graph.constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
    v = graph.transposeTensor(v, dimension: 1, withDimension: 2, name: nil)
    
    var att: MPSGraphTensor
    if (saveMemory) {
        // MEM-HACK - silly graph seems to use less peak memory
        var attRes = [MPSGraphTensor]()
        let sliceSize = 1
        for i in 0..<nHeads.intValue/sliceSize {
            let qi = graph.sliceTensor(q, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
            let ki = graph.sliceTensor(k, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
            let vi = graph.sliceTensor(v, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
            var attI = graph.matrixMultiplication(primary: qi, secondary: ki, name: nil)
            attI = graph.softMax(with: attI, axis: 3, name: nil)
            attI = graph.matrixMultiplication(primary: attI, secondary: vi, name: nil)
            attI = graph.transposeTensor(attI, dimension: 1, withDimension: 2, name: nil)
            attRes.append(attI)
        }
        att = graph.concatTensors(attRes, dimension: 2, name: nil)
    } else {
        att = graph.matrixMultiplication(primary: q, secondary: k, name: nil)
        att = graph.softMax(with: att, axis: 3, name: nil)
        att = graph.matrixMultiplication(primary: att, secondary: v, name: nil)
        att = graph.transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
    }
    att = graph.reshape(att, shape: xIn.shape!, name: nil)
    return makeLinear(graph: graph, xIn: att, name: name + ".to_out.0", outChannels: c)
}

func makeGelu(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    var x = xIn
    x = graph.multiplication(x, graph.constant(1/sqrt(2), dataType: MPSDataType.float16), name: nil)
    x = graph.erf(with: x, name: nil)
    x = graph.addition(x, graph.constant(1, dataType: MPSDataType.float16), name: nil)
    x = graph.multiplication(x, graph.constant(0.5, dataType: MPSDataType.float16), name: nil)
    return graph.multiplication(xIn, x, name: nil)
}

func makeFeedForward(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    assert(xIn.shape!.count == 3)
    let dim = xIn.shape![2]
    let dimMult = dim.intValue * 4
    let dimProj = NSNumber(value: dimMult * 2)
    let proj = makeLinear(graph: graph, xIn: xIn, name: name + ".0.proj", outChannels: dimProj)
    var x = graph.sliceTensor(proj, dimension: 2, start: 0, length: dimMult, name: nil)
    var gate = graph.sliceTensor(proj, dimension: 2, start: dimMult, length: dimMult, name: nil)
    gate = makeGelu(graph: graph, xIn: gate)
    x = graph.multiplication(x, gate, name: nil)
    return makeLinear(graph: graph, xIn: x, name: name + ".2", outChannels: dim)
}

func makeBasicTransformerBlock(graph: MPSGraph, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
    var x = xIn
    var attn1 = makeLayerNorm(graph: graph, xIn: x, name: name + ".norm1")
    attn1 = makeCrossAttention(graph: graph, xIn: attn1, name: name + ".attn1", context: nil, saveMemory: saveMemory)
    x = graph.addition(attn1, x, name: nil)
    var attn2 = makeLayerNorm(graph: graph, xIn: x, name: name + ".norm2")
    attn2 = makeCrossAttention(graph: graph, xIn: attn2, name: name + ".attn2", context: contextIn, saveMemory: saveMemory)
    x = graph.addition(attn2, x, name: nil)
    var ff = makeLayerNorm(graph: graph, xIn: x, name: name + ".norm3")
    ff = makeFeedForward(graph: graph, xIn: ff, name: name + ".ff.net")
    return graph.addition(ff, x, name: nil)
}

@available(macOS 12.3, *)
func makeSpatialTransformerBlock(graph: MPSGraph, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
    let n, h, w, c: NSNumber
    (n, h, w, c) = (xIn.shape![0], xIn.shape![1], xIn.shape![2], xIn.shape![3])
    var x = xIn
    x = makeGroupNorm(graph: graph, xIn: x, name: name + ".norm")
    x = makeConv(graph: graph, xIn: x, name: name + ".proj_in", outChannels: c, khw: 1)
    x = graph.reshape(x, shape: [n, (h.intValue * w.intValue) as NSNumber, c], name: nil)
    x = makeBasicTransformerBlock(graph: graph, xIn: x, name: name + ".transformer_blocks.0", contextIn: contextIn, saveMemory: saveMemory)
    x = graph.reshape(x, shape: [n, h, w, c], name: nil)
    x = makeConv(graph: graph, xIn: x, name: name + ".proj_out", outChannels: c, khw: 1)
    return graph.addition(x, xIn, name: nil)
}

@available(macOS 12.3, *)
func makeOutputBlock(graph: MPSGraph, xIn: MPSGraphTensor, embIn: MPSGraphTensor, condIn: MPSGraphTensor, inChannels: NSNumber, outChannels: NSNumber, dHead: NSNumber, name: String, saveMemory: Bool, spatialTransformer: Bool = true, upsample: Bool = false) -> MPSGraphTensor {
    var x = xIn
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: embIn, name: name + ".0", inChannels: inChannels, outChannels: outChannels)
    if (spatialTransformer) {
        x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".1", contextIn: condIn, saveMemory: saveMemory)
    }
    if (upsample) {
        x = makeUpsampleNearest(graph: graph, xIn: x)
        x = makeConv(graph: graph, xIn: x, name: name + (spatialTransformer ? ".2" : ".1") + ".conv", outChannels: outChannels, khw: 3)
    }
    return x
}


@available(macOS 12.3, *)
func makeUNetAnUnexpectedJourney(graph: MPSGraph, xIn: MPSGraphTensor, tembIn: MPSGraphTensor, condIn: MPSGraphTensor, name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
    let emb = makeTimeEmbed(graph: graph, xIn: tembIn, name: name + ".time_embed")
    
    var savedInputs = [MPSGraphTensor]()
    var x = xIn
    
    if (!saveMemory) {
        // need to explicitly batch to avoid shape errors later iirc
        // TODO: did we actually need this
        x = graph.broadcast(x, shape: [condIn.shape![0], x.shape![1], x.shape![2], x.shape![3]], name: nil)
    }
    
    // input blocks
    x = makeConv(graph: graph, xIn: x, name: name + ".input_blocks.0.0", outChannels: 320, khw: 3)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.1.0", inChannels: 320, outChannels: 320)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.1.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.2.0", inChannels: 320, outChannels: 320)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.2.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    // downsample
    x = makeConv(graph: graph, xIn: x, name: name + ".input_blocks.3.0.op", outChannels: 320, khw: 3, stride: 2)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.4.0", inChannels: 320, outChannels: 640)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.4.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.5.0", inChannels: 640, outChannels: 640)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.5.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    // downsample
    x = makeConv(graph: graph, xIn: x, name: name + ".input_blocks.6.0.op", outChannels: 640, khw: 3, stride: 2)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.7.0", inChannels: 640, outChannels: 1280)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.7.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.8.0", inChannels: 1280, outChannels: 1280)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".input_blocks.8.1", contextIn: condIn, saveMemory: saveMemory)
    savedInputs.append(x)
    
    // downsample
    x = makeConv(graph: graph, xIn: x, name: name + ".input_blocks.9.0.op", outChannels: 1280, khw: 3, stride: 2)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.10.0", inChannels: 1280, outChannels: 1280)
    savedInputs.append(x)
    
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".input_blocks.11.0", inChannels: 1280, outChannels: 1280)
    savedInputs.append(x)
    
    // middle blocks
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".middle_block.0", inChannels: 1280, outChannels: 1280)
    x = makeSpatialTransformerBlock(graph: graph, xIn: x, name: name + ".middle_block.1", contextIn: condIn, saveMemory: saveMemory)
    x = makeUNetResBlock(graph: graph, xIn: x, embIn: emb, name: name + ".middle_block.2", inChannels: 1280, outChannels: 1280)
    
    return savedInputs + [emb] + [x]
}

@available(macOS 12.3, *)
func makeUNetTheDesolationOfSmaug(graph: MPSGraph, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
    var savedInputs = savedInputsIn
    let condIn = savedInputs.popLast()!
    var x = savedInputs.popLast()!
    let emb = savedInputs.popLast()!
    // output blocks
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.0", saveMemory: saveMemory, spatialTransformer: false, upsample: false)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.1", saveMemory: saveMemory, spatialTransformer: false, upsample: false)
    
    // upsample
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.2", saveMemory: saveMemory, spatialTransformer: false, upsample: true)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.3", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.4", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    return savedInputs + [emb] + [x]
}

@available(macOS 12.3, *)
func makeUNetTheBattleOfTheFiveArmies(graph: MPSGraph, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> MPSGraphTensor {
    var savedInputs = savedInputsIn
    let condIn = savedInputs.popLast()!
    var x = savedInputs.popLast()!
    let emb = savedInputs.popLast()!
    // upsample
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 1920, outChannels: 1280, dHead: 160, name: name + ".output_blocks.5", saveMemory: saveMemory, spatialTransformer: true, upsample: true)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 1920, outChannels: 640, dHead: 80, name: name + ".output_blocks.6", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 1280, outChannels: 640, dHead: 80, name: name + ".output_blocks.7", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    // upsample
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 960, outChannels: 640, dHead: 80, name: name + ".output_blocks.8", saveMemory: saveMemory, spatialTransformer: true, upsample: true)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 960, outChannels: 320, dHead: 40, name: name + ".output_blocks.9", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 640, outChannels: 320, dHead: 40, name: name + ".output_blocks.10", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    x = graph.concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
    x = makeOutputBlock(graph: graph, xIn: x, embIn: emb, condIn: condIn, inChannels: 640, outChannels: 320, dHead: 40, name: name + ".output_blocks.11", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
    
    // out
    x = makeGroupNormSwish(graph: graph, xIn: x, name: "model.diffusion_model.out.0")
    return makeConv(graph: graph, xIn: x, name: "model.diffusion_model.out.2", outChannels: 4, khw: 3)
}

func makeTimeFeatures(graph: MPSGraph, tIn: MPSGraphTensor) -> MPSGraphTensor {
    var temb = graph.cast(tIn, to: MPSDataType.float32, name: "temb")
    var coeffs = loadConstant(graph: graph, name: "temb_coefficients", shape: [160], fp32: true)
    coeffs = graph.cast(coeffs, to: MPSDataType.float32, name: "coeffs")
    temb = graph.multiplication(temb, coeffs, name: nil)
    temb = graph.concatTensors([graph.cos(with: temb, name: nil), graph.sin(with: temb, name: nil)], dimension: 0, name: nil)
    temb = graph.reshape(temb, shape: [1, 320], name: nil)
    return graph.cast(temb, to: MPSDataType.float16, name: "temb fp16")
}

func makeSqrtOneMinus(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    return graph.squareRoot(with: graph.subtraction(graph.constant(1.0, dataType: MPSDataType.float16), xIn, name: nil), name: nil)
}

@available(macOS 12.3, *)
func makeDiffusionStep(graph: MPSGraph, xIn: MPSGraphTensor, etaUncondIn: MPSGraphTensor, etaCondIn: MPSGraphTensor, tIn: MPSGraphTensor, tPrevIn: MPSGraphTensor, guidanceScaleIn: MPSGraphTensor) -> MPSGraphTensor {
    
    // superconditioning
    var deltaCond = graph.multiplication(graph.subtraction(etaCondIn, etaUncondIn, name: nil), guidanceScaleIn, name: nil)
    deltaCond = graph.tanh(with: deltaCond, name: nil) // NOTE: normal SD doesn't clamp here iirc
    let eta = graph.addition(etaUncondIn, deltaCond, name: nil)
    
    // scheduler conditioning
    let alphasCumprod = loadConstant(graph: graph, name: "alphas_cumprod", shape: [1000])
    let alphaIn = graph.gatherAlongAxis(0, updates: alphasCumprod, indices: tIn, name: nil)
    let alphasCumprodPrev = graph.concatTensors([graph.constant(1, dataType: MPSDataType.float16), alphasCumprod], dimension: 0, name: nil)
    let tPrevInOffset = graph.reLU(with: graph.addition(tPrevIn, graph.constant(1, dataType: MPSDataType.int32), name: nil), name: nil)
    let alphaPrevIn = graph.gatherAlongAxis(0, updates: alphasCumprodPrev, indices: tPrevInOffset, name: nil)
    
    // scheduler step
    let deltaX0 = graph.multiplication(makeSqrtOneMinus(graph: graph, xIn: alphaIn), eta, name: nil)
    let predX0Unscaled = graph.subtraction(xIn, deltaX0, name: nil)
    let predX0 = graph.division(predX0Unscaled, graph.squareRoot(with: alphaIn, name: nil), name: nil)
    let dirX = graph.multiplication(makeSqrtOneMinus(graph: graph, xIn: alphaPrevIn), eta, name: nil)
    let xPrevBase = graph.multiplication(graph.squareRoot(with: alphaPrevIn, name: nil), predX0, name:nil)
    return graph.addition(xPrevBase, dirX, name: nil)
}


func makeTextAttention(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    let nHeads: NSNumber = 12
    let dHead: NSNumber = 64
    let c: NSNumber = 768
    var q = makeLinear(graph: graph, xIn: xIn, name: name + ".q_proj", outChannels: c)
    var k = makeLinear(graph: graph, xIn: xIn, name: name + ".k_proj", outChannels: c)
    var v = makeLinear(graph: graph, xIn: xIn, name: name + ".v_proj", outChannels: c)
    
    let n = xIn.shape![0]
    let t = xIn.shape![1]
    q = graph.reshape(q, shape: [n, t, nHeads, dHead], name: nil)
    k = graph.reshape(k, shape: [n, t, nHeads, dHead], name: nil)
    v = graph.reshape(v, shape: [n, t, nHeads, dHead], name: nil)
    
    q = graph.transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
    k = graph.transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
    v = graph.transposeTensor(v, dimension: 1, withDimension: 2, name: nil)
    
    var att = graph.matrixMultiplication(primary: q, secondary: graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
    att = graph.multiplication(att, graph.constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
    att = graph.addition(att, loadConstant(graph: graph, name: "causal_mask", shape: [1, 1, 77, 77]), name: nil)
    att = graph.softMax(with: att, axis: 3, name: nil)
    att = graph.matrixMultiplication(primary: att, secondary: v, name: nil)
    att = graph.transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
    att = graph.reshape(att, shape: [n, t, c], name: nil)
    return makeLinear(graph: graph, xIn: att, name: name + ".out_proj", outChannels: c)
}

func makeTextEncoderLayer(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = xIn
    x = makeLayerNorm(graph: graph, xIn: x, name: name + ".layer_norm1")
    x = makeTextAttention(graph: graph, xIn: x, name: name + ".self_attn")
    x = graph.addition(x, xIn, name: nil)
    let skip = x
    x = makeLayerNorm(graph: graph, xIn: x, name: name + ".layer_norm2")
    x = makeLinear(graph: graph, xIn: x, name: name + ".mlp.fc1", outChannels: 3072)
    x = makeGelu(graph: graph, xIn: x)
    x = makeLinear(graph: graph, xIn: x, name: name + ".mlp.fc2", outChannels: 768)
    return graph.addition(x, skip, name: nil)
}

func makeTextEncoder(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = xIn
    for i in 0..<12 {
        x = makeTextEncoderLayer(graph: graph, xIn: x, name: name + ".layers.\(i)")
    }
    return x
}

@available(macOS 12.3, *)
func makeTextEmbeddings(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var tokenEmbeddings = loadConstant(graph: graph, name: name + ".token_embedding.weight", shape: [1, 49408, 768])
    tokenEmbeddings = graph.broadcast(tokenEmbeddings, shape: [2, 49408, 768], name: nil)
    let positionEmbeddings = loadConstant(graph: graph, name: name + ".position_embedding.weight", shape: [1, 77, 768])
    var embeddings = graph.broadcast(graph.expandDims(xIn, axes: [2], name: nil), shape: [2, 77, 768], name: nil)
    embeddings = graph.gatherAlongAxis(1, updates: tokenEmbeddings, indices: embeddings, name: nil)
    return graph.addition(embeddings, positionEmbeddings, name: nil)
}

@available(macOS 12.3, *)
func makeTextGuidance(graph: MPSGraph, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
    var x = makeTextEmbeddings(graph: graph, xIn: xIn, name: name + ".embeddings")
    x = makeTextEncoder(graph: graph, xIn: x, name: name + ".encoder")
    return makeLayerNorm(graph: graph, xIn: x, name: name + ".final_layer_norm")
}

func makeAuxUpsampler(graph: MPSGraph, xIn: MPSGraphTensor) -> MPSGraphTensor {
    var x = xIn
    x = makeConv(graph: graph, xIn: xIn, name: "aux_output_conv", outChannels: 3, khw: 1)
    x = makeUpsampleNearest(graph: graph, xIn: x, scaleFactor: 8)
    return makeByteConverter(graph: graph, xIn: x)
}


func tensorToCGImage(data: MPSGraphTensorData) -> CGImage {
    let shape = data.shape.map{$0.intValue}
    var imageArrayCPUBytes = [UInt8](repeating: 0, count: shape.reduce(1, *))
    data.mpsndarray().readBytes(&imageArrayCPUBytes, strideBytes: nil)
    
    return CGImage(
        width: shape[2], height: shape[1],
        bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: shape[2]*shape[3],
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo:  CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: CGDataProvider(data: NSData(bytes: &imageArrayCPUBytes,
                                              length: imageArrayCPUBytes.count))!,
        decode: nil,
        shouldInterpolate: true,
        intent: CGColorRenderingIntent.defaultIntent)!
}
