//
//  Scheduler.swift
//  
//
//  Created by Guillermo Cique Fernández on 13/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class Scheduler {
    let count: Int
    private let timesteps: [Int]
    let timestepSize: Int
    var timestepsData: MPSGraphTensorData {
        let data = timesteps.map { Int32($0) }.withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(
            device: device,
            data: data,
            shape: [NSNumber(value: timesteps.count)],
            dataType: MPSDataType.int32
        )
    }
    
    private let device: MPSGraphDevice
    private let graph: MPSGraph
    private let timestepIn: MPSGraphTensor
    private let tembOut: MPSGraphTensor
    
    init(synchronize: Bool, modelLocation: URL, device: MPSGraphDevice, steps: Int) {
        self.device = device
        count = steps
        
        timestepSize = 1000 / steps
        timesteps = Array<Int>(stride(from: 1, to: 1000, by: timestepSize))
        graph = MPSGraph(synchronize: synchronize)
        timestepIn = graph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        tembOut = graph.makeTimeFeatures(at: modelLocation, tIn: timestepIn)
    }
    
    func timesteps(strength: Float?) -> [Int] {
        guard let strength else { return timesteps.reversed() }
        let startStep = Int(Float(count) * strength)
        return timesteps[0..<startStep].reversed()
    }
    
    func run(with queue: MTLCommandQueue, timestep: Int) -> MPSGraphTensorData {
        let timestepData = [Int32(timestep)].withUnsafeBufferPointer { Data(buffer: $0) }
        let data = MPSGraphTensorData(device: device, data: timestepData, shape: [1], dataType: MPSDataType.int32)
        return graph.run(
            with: queue,
            feeds: [timestepIn: data],
            targetTensors: [tembOut],
            targetOperations: nil
        )[tembOut]!
    }
}

extension MPSGraph {
    func makeTimeFeatures(at folder: URL, tIn: MPSGraphTensor) -> MPSGraphTensor {
        var temb = cast(tIn, to: MPSDataType.float32, name: "temb")
        var coeffs = loadConstant(at: folder, name: "temb_coefficients", shape: [160], fp32: true)
        coeffs = cast(coeffs, to: MPSDataType.float32, name: "coeffs")
        temb = multiplication(temb, coeffs, name: nil)
        temb = concatTensors([cos(with: temb, name: nil), sin(with: temb, name: nil)], dimension: 0, name: nil)
        temb = reshape(temb, shape: [1, 320], name: nil)
        return cast(temb, to: MPSDataType.float16, name: "temb fp16")
    }
}
