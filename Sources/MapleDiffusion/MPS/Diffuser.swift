//
//  Diffuser.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class Diffuser {
    private let device: MPSGraphDevice
    private let graph: MPSGraph
    private let xIn: MPSGraphTensor
    private let etaUncondIn: MPSGraphTensor
    private let etaCondIn: MPSGraphTensor
    private let timestepIn: MPSGraphTensor
    private let timestepSizeIn: MPSGraphTensor
    private let guidanceScaleIn: MPSGraphTensor
    private let out: MPSGraphTensor
    private let auxOut: MPSGraphTensor
    
    init(synchronize: Bool, device: MPSGraphDevice, modelLocation: URL, shape: [NSNumber]) {
        self.device = device
        graph = MPSGraph(synchronize: synchronize)
        xIn = graph.placeholder(shape: shape, dataType: MPSDataType.float16, name: nil)
        etaUncondIn = graph.placeholder(shape: shape, dataType: MPSDataType.float16, name: nil)
        etaCondIn = graph.placeholder(shape: shape, dataType: MPSDataType.float16, name: nil)
        timestepIn = graph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        timestepSizeIn = graph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        guidanceScaleIn = graph.placeholder(shape: [1], dataType: MPSDataType.float32, name: nil)
        out = graph.makeDiffusionStep(
            at: modelLocation,
            xIn: xIn,
            etaUncondIn: etaUncondIn,
            etaCondIn: etaCondIn,
            timestepIn: timestepIn,
            timestepSizeIn: timestepSizeIn,
            guidanceScaleIn: graph.cast(guidanceScaleIn, to: MPSDataType.float16, name: "this string must not be the empty string")
        )
        auxOut = graph.makeAuxUpsampler(at: modelLocation, xIn: out)
    }
    
    func run(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        timestep: Int,
        timestepSize: Int,
        etaUncond: MPSGraphTensorData,
        etaCond: MPSGraphTensorData,
        guidanceScale: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData?) {
        let timestepData = timestep.tensorData(device: device)
        let timestepSizeData = timestepSize.tensorData(device: device)
        let outputs = graph.run(
            with: queue,
            feeds: [
                xIn: latent,
                etaUncondIn: etaUncond,
                etaCondIn: etaCond,
                timestepIn: timestepData,
                timestepSizeIn: timestepSizeData,
                guidanceScaleIn: guidanceScale
            ],
            targetTensors: [out, auxOut],
            targetOperations: nil
        )
        return (outputs[out]!, outputs[auxOut])
    }
}

fileprivate extension MPSGraph {
    func makeDiffusionStep(
        at folder: URL,
        xIn: MPSGraphTensor,
        etaUncondIn: MPSGraphTensor,
        etaCondIn: MPSGraphTensor,
        timestepIn: MPSGraphTensor,
        timestepSizeIn: MPSGraphTensor,
        guidanceScaleIn: MPSGraphTensor
    ) -> MPSGraphTensor {
        
        // superconditioning
        var deltaCond = multiplication(subtraction(etaCondIn, etaUncondIn, name: nil), guidanceScaleIn, name: nil)
        deltaCond = tanh(with: deltaCond, name: nil) // NOTE: normal SD doesn't clamp here iirc
        let eta = addition(etaUncondIn, deltaCond, name: nil)
        
        // scheduler conditioning
        let alphasCumprod = loadConstant(at: folder, name: "alphas_cumprod", shape: [1000])
        let alphaIn = gatherAlongAxis(0, updates: alphasCumprod, indices: timestepIn, name: nil)
        let prevTimestep = maximum(
            constant(0, dataType: MPSDataType.int32),
            subtraction(timestepIn, timestepSizeIn, name: nil),
            name: nil
        )
        let alphaPrevIn = gatherAlongAxis(0, updates: alphasCumprod, indices: prevTimestep, name: nil)
        
        // scheduler step
        let deltaX0 = multiplication(squareRootOfOneMinus(alphaIn), eta, name: nil)
        let predX0Unscaled = subtraction(xIn, deltaX0, name: nil)
        let predX0 = division(predX0Unscaled, squareRoot(with: alphaIn, name: nil), name: nil)
        let dirX = multiplication(squareRootOfOneMinus(alphaPrevIn), eta, name: nil)
        let xPrevBase = multiplication(squareRoot(with: alphaPrevIn, name: nil), predX0, name:nil)
        return addition(xPrevBase, dirX, name: nil)
    }
    
    func makeAuxUpsampler(at folder: URL, xIn: MPSGraphTensor) -> MPSGraphTensor {
        var x = xIn
        x = makeConv(at: folder, xIn: xIn, name: "aux_output_conv", outChannels: 3, khw: 1)
        x = upsampleNearest(xIn: x, scaleFactor: 8)
        return makeByteConverter(xIn: x)
    }
}
