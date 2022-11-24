//
//  MPSGraph+Transformer.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func makeSpatialTransformerBlock(at folder: URL, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
        let n, h, w, c: NSNumber
        (n, h, w, c) = (xIn.shape![0], xIn.shape![1], xIn.shape![2], xIn.shape![3])
        var x = xIn
        x = makeGroupNorm(at: folder, xIn: x, name: name + ".norm")
        x = makeConv(at: folder, xIn: x, name: name + ".proj_in", outChannels: c, khw: 1)
        x = reshape(x, shape: [n, (h.intValue * w.intValue) as NSNumber, c], name: nil)
        x = makeBasicTransformerBlock(at: folder, xIn: x, name: name + ".transformer_blocks.0", contextIn: contextIn, saveMemory: saveMemory)
        x = reshape(x, shape: [n, h, w, c], name: nil)
        x = makeConv(at: folder, xIn: x, name: name + ".proj_out", outChannels: c, khw: 1)
        return addition(x, xIn, name: nil)
    }
    
    fileprivate func makeBasicTransformerBlock(at folder: URL, xIn: MPSGraphTensor, name: String, contextIn: MPSGraphTensor, saveMemory: Bool) -> MPSGraphTensor {
        var x = xIn
        var attn1 = makeLayerNorm(at: folder, xIn: x, name: name + ".norm1")
        attn1 = makeCrossAttention(at: folder, xIn: attn1, name: name + ".attn1", context: nil, saveMemory: saveMemory)
        x = addition(attn1, x, name: nil)
        var attn2 = makeLayerNorm(at: folder, xIn: x, name: name + ".norm2")
        attn2 = makeCrossAttention(at: folder, xIn: attn2, name: name + ".attn2", context: contextIn, saveMemory: saveMemory)
        x = addition(attn2, x, name: nil)
        var ff = makeLayerNorm(at: folder, xIn: x, name: name + ".norm3")
        ff = makeFeedForward(at: folder, xIn: ff, name: name + ".ff.net")
        return addition(ff, x, name: nil)
    }
    
    fileprivate func makeFeedForward(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        assert(xIn.shape!.count == 3)
        let dim = xIn.shape![2]
        let dimMult = dim.intValue * 4
        let dimProj = NSNumber(value: dimMult * 2)
        let proj = makeLinear(at: folder, xIn: xIn, name: name + ".0.proj", outChannels: dimProj)
        var x = sliceTensor(proj, dimension: 2, start: 0, length: dimMult, name: nil)
        var gate = sliceTensor(proj, dimension: 2, start: dimMult, length: dimMult, name: nil)
        gate = gelu(gate)
        x = multiplication(x, gate, name: nil)
        return makeLinear(at: folder, xIn: x, name: name + ".2", outChannels: dim)
    }
    
    fileprivate func makeCrossAttention(at folder: URL, xIn: MPSGraphTensor, name: String, context: MPSGraphTensor?, saveMemory: Bool) -> MPSGraphTensor {
        let c = xIn.shape![2]
        let (nHeads, dHead) = (NSNumber(8), NSNumber(value: c.intValue / 8))
        var q = makeLinear(at: folder, xIn: xIn, name: name + ".to_q", outChannels: c, bias: false)
        let context = context ?? xIn
        var k = makeLinear(at: folder, xIn: context, name: name + ".to_k", outChannels: c, bias: false)
        var v = makeLinear(at: folder, xIn: context, name: name + ".to_v", outChannels: c, bias: false)
        let n = xIn.shape![0]
        let hw = xIn.shape![1]
        let t = context.shape![1]
        q = reshape(q, shape: [n, hw, nHeads, dHead], name: nil)
        k = reshape(k, shape: [n, t, nHeads, dHead], name: nil)
        v = reshape(v, shape: [n, t, nHeads, dHead], name: nil)
        
        q = transposeTensor(q, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 1, withDimension: 2, name: nil)
        k = transposeTensor(k, dimension: 2, withDimension: 3, name: nil)
        k = multiplication(k, constant(1.0 / sqrt(dHead.doubleValue), dataType: MPSDataType.float16), name: nil)
        v = transposeTensor(v, dimension: 1, withDimension: 2, name: nil)
        
        var att: MPSGraphTensor
        if (saveMemory) {
            // MEM-HACK - silly graph seems to use less peak memory
            var attRes = [MPSGraphTensor]()
            let sliceSize = 1
            for i in 0..<nHeads.intValue/sliceSize {
                let qi = sliceTensor(q, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
                let ki = sliceTensor(k, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
                let vi = sliceTensor(v, dimension: 1, start: i*sliceSize, length: sliceSize, name: nil)
                var attI = matrixMultiplication(primary: qi, secondary: ki, name: nil)
                attI = softMax(with: attI, axis: 3, name: nil)
                attI = matrixMultiplication(primary: attI, secondary: vi, name: nil)
                attI = transposeTensor(attI, dimension: 1, withDimension: 2, name: nil)
                attRes.append(attI)
            }
            att = concatTensors(attRes, dimension: 2, name: nil)
        } else {
            att = matrixMultiplication(primary: q, secondary: k, name: nil)
            att = softMax(with: att, axis: 3, name: nil)
            att = matrixMultiplication(primary: att, secondary: v, name: nil)
            att = transposeTensor(att, dimension: 1, withDimension: 2, name: nil)
        }
        att = reshape(att, shape: xIn.shape!, name: nil)
        return makeLinear(at: folder, xIn: att, name: name + ".to_out.0", outChannels: c)
    }
}
