//
//  MPSGraph.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {
    convenience init(synchronize: Bool) {
        self.init()
        options = synchronize ? MPSGraphOptions.synchronizeResults : .none
    }
    
    func loadConstant(at folder: URL, name: String, shape: [NSNumber], fp32: Bool = false) -> MPSGraphTensor {
        let numels = shape.map({$0.intValue}).reduce(1, *)
        //    let fileUrl: URL = Bundle.main.url(forResource: "bins/" + name + (fp32 ? "_fp32" : ""), withExtension: ".bin")!
        let fileUrl: URL = folder.appendingPathComponent(name + (fp32 ? "_fp32" : "")).appendingPathExtension("bin")
        let data: Data = try! Data(contentsOf: fileUrl, options: Data.ReadingOptions.alwaysMapped)
        let expectedCount = numels * (fp32 ? 4 : 2)
        assert(data.count == expectedCount, "Mismatch between byte count of data \(data.count) and expected size \(expectedCount) for \(numels) els in \(fileUrl)")
        return constant(data, shape: shape, dataType: fp32 ? MPSDataType.float32 : MPSDataType.float16)
    }
    
    
    func makeConv(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, khw: NSNumber, stride: Int = 1, bias: Bool = true) -> MPSGraphTensor {
        let w = loadConstant(at: folder, name: name + ".weight", shape: [outChannels, xIn.shape![3], khw, khw])
        let p: Int = khw.intValue / 2;
        let convDesc = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: p,
            paddingRight: p,
            paddingTop: p,
            paddingBottom: p,
            paddingStyle: MPSGraphPaddingStyle.explicit,
            dataLayout: MPSGraphTensorNamedDataLayout.NHWC,
            weightsLayout: MPSGraphTensorNamedDataLayout.OIHW
        )!
        let conv = convolution2D(xIn, weights: w, descriptor: convDesc, name: nil)
        if (bias) {
            let b = loadConstant(at: folder, name: name + ".bias", shape: [1, 1, 1, outChannels])
            return addition(conv, b, name: nil)
        }
        return conv
    }
    
    func makeLinear(at folder: URL, xIn: MPSGraphTensor, name: String, outChannels: NSNumber, bias: Bool = true) -> MPSGraphTensor {
        if (xIn.shape!.count == 2) {
            var x = reshape(xIn, shape: [xIn.shape![0], 1, 1, xIn.shape![1]], name: nil)
            x = makeConv(at: folder, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
            return reshape(x, shape: [xIn.shape![0], outChannels], name: nil)
        }
        var x = reshape(xIn, shape: [xIn.shape![0], 1, xIn.shape![1], xIn.shape![2]], name: nil)
        x = makeConv(at: folder, xIn: x, name: name, outChannels: outChannels, khw: 1, bias: bias)
        return reshape(x, shape: [xIn.shape![0], xIn.shape![1], outChannels], name: nil)
    }
    
    
    func makeLayerNorm(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        assert(xIn.shape!.count == 3, "layernorm requires NTC")
        let gamma = loadConstant(at: folder, name: name + ".weight", shape: [1, 1, xIn.shape![2]])
        let beta = loadConstant(at: folder, name: name + ".bias", shape: [1,  1, xIn.shape![2]])
        let mean = mean(of: xIn, axes: [2], name: nil)
        let variance = variance(of: xIn, axes: [2], name: nil)
        let x = normalize(xIn, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        return reshape(x, shape: xIn.shape!, name: nil)
    }
    
    func makeGroupNorm(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        if (xIn.shape!.count == 3) {
            x = expandDims(x, axes: [1], name: nil)
        }
        let shape = x.shape!
        let nGroups: NSNumber = 32
        let nGrouped: NSNumber = shape[3].floatValue / nGroups.floatValue as NSNumber
        let gamma = loadConstant(at: folder, name: name + ".weight", shape: [1, 1, 1, nGroups, nGrouped])
        let beta = loadConstant(at: folder, name: name + ".bias", shape: [1, 1, 1, nGroups, nGrouped])
        x = reshape(x, shape: [shape[0], shape[1], shape[2], nGroups, nGrouped], name: nil)
        let mean = mean(of: x, axes: [1, 2, 4], name: nil)
        let variance = variance(of: x, axes: [1, 2, 4], name: nil)
        x = normalize(x, mean: mean, variance: variance, gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
        return reshape(x, shape: xIn.shape!, name: nil)
    }

    func makeGroupNormSwish(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        return swish(makeGroupNorm(at: folder, xIn: xIn, name: name))
    }
    
    func makeAuxUpsampler(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        x = makeConv(at: folder, xIn: xIn, name: "aux_output_conv", outChannels: 3, khw: 1)
        x = upsampleNearest(xIn: x, scaleFactor: 8)
        return makeByteConverter(xIn: x)
    }
    
    func makeByteConverter(xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        x = clamp(x, min: constant(0, shape: [1], dataType: MPSDataType.float16), max: constant(1.0, shape: [1], dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(255, shape: [1], dataType: MPSDataType.float16), name: nil)
        x = round(with: x, name: nil)
        x = cast(x, to: MPSDataType.uInt8, name: "cast to uint8 rgba")
        let alpha = constant(255, shape: [1,  x.shape![1], x.shape![2], 1], dataType: MPSDataType.uInt8)
        return concatTensors([x, alpha], dimension: 3, name: nil)
    }
    
    func makeTimeFeatures(at folder: URL, tIn: MPSGraphTensor) -> MPSGraphTensor {
        var temb = cast(tIn, to: MPSDataType.float32, name: "temb")
        var coeffs = loadConstant(at: folder, name: "temb_coefficients", shape: [160], fp32: true)
        coeffs = cast(coeffs, to: MPSDataType.float32, name: "coeffs")
        temb = multiplication(temb, coeffs, name: nil)
        temb = concatTensors([cos(with: temb, name: nil), sin(with: temb, name: nil)], dimension: 0, name: nil)
        temb = reshape(temb, shape: [1, 320], name: nil)
        return cast(temb, to: MPSDataType.float16, name: "temb fp16")
    }

    func makeDiffusionStep(at folder: URL, xIn: MPSGraphTensor, etaUncondIn: MPSGraphTensor, etaCondIn: MPSGraphTensor, tIn: MPSGraphTensor, tPrevIn: MPSGraphTensor, guidanceScaleIn: MPSGraphTensor) -> MPSGraphTensor {
        
        // superconditioning
        var deltaCond = multiplication(subtraction(etaCondIn, etaUncondIn, name: nil), guidanceScaleIn, name: nil)
        deltaCond = tanh(with: deltaCond, name: nil) // NOTE: normal SD doesn't clamp here iirc
        let eta = addition(etaUncondIn, deltaCond, name: nil)
        
        // scheduler conditioning
        let alphasCumprod = loadConstant(at: folder, name: "alphas_cumprod", shape: [1000])
        let alphaIn = gatherAlongAxis(0, updates: alphasCumprod, indices: tIn, name: nil)
        let alphasCumprodPrev = concatTensors([constant(1, dataType: MPSDataType.float16), alphasCumprod], dimension: 0, name: nil)
        let tPrevInOffset = reLU(with: addition(tPrevIn, constant(1, dataType: MPSDataType.int32), name: nil), name: nil)
        let alphaPrevIn = gatherAlongAxis(0, updates: alphasCumprodPrev, indices: tPrevInOffset, name: nil)
        
        // scheduler step
        let deltaX0 = multiplication(squareRootOfOneMinus(alphaIn), eta, name: nil)
        let predX0Unscaled = subtraction(xIn, deltaX0, name: nil)
        let predX0 = division(predX0Unscaled, squareRoot(with: alphaIn, name: nil), name: nil)
        let dirX = multiplication(squareRootOfOneMinus(alphaPrevIn), eta, name: nil)
        let xPrevBase = multiplication(squareRoot(with: alphaPrevIn, name: nil), predX0, name:nil)
        return addition(xPrevBase, dirX, name: nil)
    }
}

// MARK: Operations

extension MPSGraph {
    func swish(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        return multiplication(tensor, sigmoid(with: tensor, name: nil), name: nil)
    }
    
    func upsampleNearest(xIn: MPSGraphTensor, scaleFactor: Int = 2) -> MPSGraphTensor {
        return resize(
            xIn,
            size: [
                NSNumber(value:xIn.shape![1].intValue * scaleFactor),
                NSNumber(value:xIn.shape![2].intValue * scaleFactor)
            ],
            mode: MPSGraphResizeMode.nearest,
            centerResult: true,
            alignCorners: false,
            layout: MPSGraphTensorNamedDataLayout.NHWC,
            name: nil
        )
    }
    
    func squareRootOfOneMinus(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        return squareRoot(with: subtraction(constant(1.0, dataType: MPSDataType.float16), tensor, name: nil), name: nil)
    }
    
    // Gaussian Error Linear Units
    func gelu(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
        var x = tensor
        x = multiplication(x, constant(1/sqrt(2), dataType: MPSDataType.float16), name: nil)
        x = erf(with: x, name: nil)
        x = addition(x, constant(1, dataType: MPSDataType.float16), name: nil)
        x = multiplication(x, constant(0.5, dataType: MPSDataType.float16), name: nil)
        return multiplication(tensor, x, name: nil)
    }
}
