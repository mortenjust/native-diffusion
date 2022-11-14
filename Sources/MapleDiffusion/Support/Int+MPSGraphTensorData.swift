//
//  File.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension Int {
    func tensorData(device: MPSGraphDevice) -> MPSGraphTensorData {
        let data = [Int32(self)].withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: device, data: data, shape: [1], dataType: MPSDataType.int32)
    }
}
