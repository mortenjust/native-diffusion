//
//  Encoder.swift
//
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class Encoder {
    let device: MPSGraphDevice
    private let graph: MPSGraph
    private let encoderIn: MPSGraphTensor!
    private let encoderOut: MPSGraphTensor!
    private let noise: MPSGraphTensor!
    private let gaussianOut: MPSGraphTensor!
    private let scaled: MPSGraphTensor!
    private let stochasticEncode: MPSGraphTensor!
    private let stepIn: MPSGraphTensor!
    private let timestepsIn: MPSGraphTensor!
    
    init(synchronize: Bool,
         modelLocation: URL,
         device: MPSGraphDevice,
         inputShape: [NSNumber],
         outputShape: [NSNumber],
         timestepsShape: [NSNumber],
         seed: Int
    ) {
        self.device = device
        graph = MPSGraph(synchronize: synchronize)
        
        encoderIn = graph.placeholder(shape: inputShape, dataType: MPSDataType.uInt8, name: nil)
        encoderOut = graph.makeEncoder(at: modelLocation, xIn: encoderIn)
        
        noise = graph.randomTensor(
            withShape: outputShape,
            descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!,
            seed: seed,
            name: nil
        )
        gaussianOut = graph.diagonalGaussianDistribution(encoderOut, noise: noise)
        scaled = graph.multiplication(gaussianOut, graph.constant(0.18215, dataType: MPSDataType.float16), name: "rescale")
        
        stepIn = graph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        timestepsIn = graph.placeholder(shape: timestepsShape, dataType: MPSDataType.int32, name: nil)
        stochasticEncode = graph.stochasticEncode(at: modelLocation, stepIn: stepIn, timestepsIn: timestepsIn, imageIn: scaled, noiseIn: noise)
    }
    
    func run(with queue: MTLCommandQueue, image: MPSGraphTensorData, step: Int, timesteps: MPSGraphTensorData) -> MPSGraphTensorData {
        let stepData = step.tensorData(device: device)
        
        return graph.run(
            with: queue,
            feeds: [
                encoderIn: image,
                stepIn: stepData,
                timestepsIn: timesteps
            ], targetTensors: [
                noise, encoderOut, gaussianOut, scaled, stochasticEncode
            ], targetOperations: nil
        )[stochasticEncode]!
    }
}

extension MPSGraph {
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
}
