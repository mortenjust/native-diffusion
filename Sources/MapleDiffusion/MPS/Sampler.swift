//
//  Sampler.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class Sampler {
    let synchronize: Bool
    let modelLocation: URL
    let saveMemory: Bool
    let device: MPSGraphDevice
    let shape: [NSNumber]
    
    
    
    init(synchronize: Bool, modelLocation: URL, saveMemory: Bool, device: MPSGraphDevice, shape: [NSNumber]) {
        self.synchronize = synchronize
        self.modelLocation = modelLocation
        self.saveMemory = saveMemory
        self.shape = shape
        self.device = device
    }
    
    func sample(with queue: MTLCommandQueue) {
        
    }
}
