//
//  MPSGraph+Decoder.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func makeDecoder(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        let name = "first_stage_model.decoder"
        x = multiplication(x, constant(1 / 0.18215, dataType: MPSDataType.float16), name: "rescale")
        x = makeConv(at: folder, xIn: x, name: "first_stage_model.post_quant_conv", outChannels: 4, khw: 1)
        x = makeConv(at: folder, xIn: x, name: name + ".conv_in", outChannels: 512, khw: 3)
        
        // middle
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_1", outChannels: 512)
        x = makeCoderAttention(at: folder, xIn: x, name: name + ".mid.attn_1")
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_2", outChannels: 512)
        
        // block 3
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.1", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.3.block.2", outChannels: 512)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.3.upsample.conv", outChannels: 512, khw: 3)
        
        // block 2
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.1", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.2.block.2", outChannels: 512)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.2.upsample.conv", outChannels: 512, khw: 3)
        
        // block 1
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.0", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.1", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.1.block.2", outChannels: 256)
        x = upsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".up.1.upsample.conv", outChannels: 256, khw: 3)
        
        // block 0
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.0", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.1", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".up.0.block.2", outChannels: 128)
        
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm_out")
        x = makeConv(at: folder, xIn: x, name: name + ".conv_out", outChannels: 3, khw: 3)
        x = addition(x, constant(1.0, dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(0.5, dataType: MPSDataType.float16), name: nil)
        return makeByteConverter(xIn: x)
    }
    
    func makeEncoder(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        // Split into RBGA
        let xParts = split(x, numSplits: 4, axis: 2, name: nil)
        // Drop alpha channel
        x = concatTensors(xParts.dropLast(), dimension: 2, name: nil)
        x = cast(x, to: .float16, name: nil)
        x = division(x, constant(255.0, shape: [1], dataType: .float16), name: nil)
        x = expandDims(x, axis: 0, name: nil)
        x = multiplication(x, constant(2.0, shape: [1], dataType: .float16), name: nil)
        x = subtraction(x, constant(1.0, shape: [1], dataType: .float16), name: nil)
        
        let name = "first_stage_model.encoder"
        x = makeConv(at: folder, xIn: x, name: name + ".conv_in", outChannels: 128, khw: 3)
        
        // block 0
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.0.block.0", outChannels: 128)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.0.block.1", outChannels: 128)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.0.downsample.conv", outChannels: 128, khw: 3)
        
        // block 1
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.1.block.0", outChannels: 256)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.1.block.1", outChannels: 256)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.1.downsample.conv", outChannels: 256, khw: 3)
        
        // block 2
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.2.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.2.block.1", outChannels: 512)
        x = downsampleNearest(xIn: x)
        x = makeConv(at: folder, xIn: x, name: name + ".down.2.downsample.conv", outChannels: 512, khw: 3)
        
        // block 3
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.3.block.0", outChannels: 512)
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".down.3.block.1", outChannels: 512)
        
        // middle
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_1", outChannels: 512)
        x = makeCoderAttention(at: folder, xIn: x, name: name + ".mid.attn_1")
        x = makeCoderResBlock(at: folder, xIn: x, name: name + ".mid.block_2", outChannels: 512)
        
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm_out")
        x = makeConv(at: folder, xIn: x, name: name + ".conv_out", outChannels: 8, khw: 3)
        
        return makeConv(at: folder, xIn: x, name: "first_stage_model.quant_conv", outChannels: 8, khw: 1)
    }
    
    fileprivate func makeCoderResBlock(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber) -> MPSGraphTensor {
        var x = xIn
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm1")
        x = makeConv(at: folder, xIn: x, name: name + ".conv1", outChannels: outChannels, khw: 3)
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".norm2")
        x = makeConv(at: folder, xIn: x, name: name + ".conv2", outChannels: outChannels, khw: 3)
        if (xIn.shape![3] != outChannels) {
            let ninShortcut = makeConv(at: folder, xIn: xIn, name: name + ".nin_shortcut", outChannels: outChannels, khw: 1)
            return addition(x, ninShortcut, name: "skip")
        }
        return addition(x, xIn, name: "skip")
    }

    fileprivate func makeCoderAttention(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = makeGroupNorm(at: folder, xIn: xIn, name: name + ".norm")
        let c = x.shape![3]
        x = reshape(x, shape: [x.shape![0], NSNumber(value:x.shape![1].intValue * x.shape![2].intValue), c], name: nil)
        let q = makeLinear(at: folder, xIn: x, name: name + ".q", outChannels: c, bias: false)
        var k = makeLinear(at: folder, xIn: x, name: name + ".k", outChannels: c, bias: false)
        k = multiplication(k, constant(1.0 / sqrt(c.doubleValue), dataType: MPSDataType.float16), name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        let v = makeLinear(at: folder, xIn: x, name: name + ".v", outChannels: c, bias: false)
        var att = matrixMultiplication(primary: q, secondary: k, name: nil)
        att = softMax(with: att, axis: 2, name: nil)
        att = matrixMultiplication(primary: att, secondary: v, name: nil)
        x = makeLinear(at: folder, xIn: att, name: name + ".proj_out", outChannels: c)
        x = reshape(x, shape: xIn.shape!, name: nil)
        return addition(x, xIn, name: nil)
    }
}
