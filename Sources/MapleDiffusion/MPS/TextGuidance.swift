//
//  TextGuidance.swift
//  
//
//  Created by Guillermo Cique Fernández on 13/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class TextGuidance {
    private let device: MPSGraphDevice
    private let tokenizer: BPETokenizer
    private let executable: MPSGraphExecutable
    
    init(synchronize: Bool, modelLocation: URL, device: MPSGraphDevice) {
        self.device = device
        self.tokenizer = BPETokenizer(modelLocation: modelLocation)
        
        let graph = MPSGraph(synchronize: synchronize)
        let textGuidanceIn = graph.placeholder(shape: [2, 77], dataType: MPSDataType.int32, name: nil)
        let textGuidanceOut = graph.makeTextGuidance(at: modelLocation, xIn: textGuidanceIn, name: "cond_stage_model.transformer.text_model")
        let textGuidanceOut0 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 0, length: 1, name: nil)
        let textGuidanceOut1 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 1, length: 1, name: nil)
        self.executable = graph.compile(
            with: device,
            feeds: [
                textGuidanceIn: MPSGraphShapedType(shape: textGuidanceIn.shape, dataType: MPSDataType.int32)
            ],
            targetTensors: [textGuidanceOut0, textGuidanceOut1],
            targetOperations: nil,
            compilationDescriptor: nil
        )
    }
    
    func run(with queue: MTLCommandQueue, prompt: String, negativePrompt: String) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let baseTokens = tokenizer.encode(s: negativePrompt)
        let tokens = tokenizer.encode(s: prompt)
        
        let data = (baseTokens + tokens).map {Int32($0)}
            .withUnsafeBufferPointer { Data(buffer: $0) }
        let tensorData = MPSGraphTensorData(device: device, data: data, shape: [2, 77], dataType: MPSDataType.int32)
        let res = executable.run(with: queue, inputs: [tensorData], results: nil, executionDescriptor: nil)
        return (res[0], res[1])
    }
}

fileprivate extension MPSGraph {
    func makeTextGuidance(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = makeTextEmbeddings(at: folder, xIn: xIn, name: name + ".embeddings")
        x = makeTextEncoder(at: folder, xIn: x, name: name + ".encoder")
        return makeLayerNorm(at: folder, xIn: x, name: name + ".final_layer_norm")
    }
    
    func makeTextEmbeddings(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var tokenEmbeddings = loadConstant(at: folder, name: name + ".token_embedding.weight", shape: [1, 49408, 768])
        tokenEmbeddings = broadcast(tokenEmbeddings, shape: [2, 49408, 768], name: nil)
        let positionEmbeddings = loadConstant(at: folder, name: name + ".position_embedding.weight", shape: [1, 77, 768])
        var embeddings = broadcast(expandDims(xIn, axes: [2], name: nil), shape: [2, 77, 768], name: nil)
        embeddings = gatherAlongAxis(1, updates: tokenEmbeddings, indices: embeddings, name: nil)
        return addition(embeddings, positionEmbeddings, name: nil)
    }
    
    func makeTextAttention(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let nHeads: NSNumber = 12
        let dHead: NSNumber = 64
        let c: NSNumber = 768
        var q = makeLinear(at: folder, xIn: xIn, name: name + ".q_proj", outChannels: c)
        var k = makeLinear(at: folder, xIn: xIn, name: name + ".k_proj", outChannels: c)
        var v = makeLinear(at: folder, xIn: xIn, name: name + ".v_proj", outChannels: c)
        
        let n = xIn.shape![0]
        let t = xIn.shape![1]
        q = reshape(q, shape: [n, t, nHeads, dHead], name: nil)
        k = reshape(k, shape: [n, t, nHeads, dHead], name: nil)
        v = reshape(v, shape: [n, t, nHeads, dHead], name: nil)
        
        q = transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        v = transposeTensor(v, dimension: 1, withDimension: 2, name: nil)
        
        var att = matrixMultiplication(primary: q, secondary: transposeTensor(k, dimension: 2, withDimension: 3, name: nil), name: nil)
        att = multiplication(att, constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
        att = addition(att, loadConstant(at: folder, name: "causal_mask", shape: [1, 1, 77, 77]), name: nil)
        att = softMax(with: att, axis: 3, name: nil)
        att = matrixMultiplication(primary: att, secondary: v, name: nil)
        att = transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
        att = reshape(att, shape: [n, t, c], name: nil)
        return makeLinear(at: folder, xIn: att, name: name + ".out_proj", outChannels: c)
    }

    func makeTextEncoderLayer(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        x = makeLayerNorm(at: folder, xIn: x, name: name + ".layer_norm1")
        x = makeTextAttention(at: folder, xIn: x, name: name + ".self_attn")
        x = addition(x, xIn, name: nil)
        let skip = x
        x = makeLayerNorm(at: folder, xIn: x, name: name + ".layer_norm2")
        x = makeLinear(at: folder, xIn: x, name: name + ".mlp.fc1", outChannels: 3072)
        x = gelu(x)
        x = makeLinear(at: folder, xIn: x, name: name + ".mlp.fc2", outChannels: 768)
        return addition(x, skip, name: nil)
    }

    func makeTextEncoder(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        for i in 0..<12 {
            x = makeTextEncoderLayer(at: folder, xIn: x, name: name + ".layers.\(i)")
        }
        return x
    }
}
