//
//  Decoder.swift
//
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class Decoder {
    private let graph: MPSGraph
    private let input: MPSGraphTensor
    private let output: MPSGraphTensor
    
    init(synchronize: Bool, modelLocation: URL, device: MPSGraphDevice, shape: [NSNumber]) {
        graph = MPSGraph(synchronize: synchronize)
        input = graph.placeholder(shape: shape, dataType: MPSDataType.float16, name: nil)
        output = graph.makeDecoder(at: modelLocation, xIn: input)
    }
    
    func run(with queue: MTLCommandQueue, xIn: MPSGraphTensorData) -> MPSGraphTensorData {
        return graph.run(
            with: queue,
            feeds: [input: xIn],
            targetTensors: [output],
            targetOperations: nil
        )[output]!
    }
}

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
}
