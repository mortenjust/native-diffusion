/**
 
 This is the package's wrapper for the `MapleDiffusion` class. It adds a few convenient ways to use Ollin's class
 
 */

import Foundation



public class Diffusion {
    
    let mapleDiffusion : MapleDiffusion
    
    public var width : NSNumber { mapleDiffusion.width }
    public var height : NSNumber { mapleDiffusion.height }
    
    public init(saveMemoryButBeSlower: Bool = false, modelFolder folder:URL) {
        modelFolder = folder
        
        self.mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: saveMemoryButBeSlower)
    }
    
    public func initModels(completion: (Float, String)->()) {
        mapleDiffusion.initModels { progress, stage in
            completion(progress, stage)
            // update publisher
        }
    }
    
    
    public func generate(prompt: String,
                         negativePrompt: String,
                         seed: Int,
                         steps: Int,
                         guidanceScale: Float,
                         completion: @escaping (CGImage?, Float, String)->()) {
        
        mapleDiffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale) { cgImage, progress, stage in
            
            // update publisher
            completion(cgImage, progress, stage)
        }
        
    }
    
    // publisher version of generate
    
    
        
}
