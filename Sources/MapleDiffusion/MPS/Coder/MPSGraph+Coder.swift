//
//  MPSGraph+Coder.swift
//
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    func makeCoderResBlock(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber) -> MPSGraphTensor {
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

    func makeCoderAttention(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
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
