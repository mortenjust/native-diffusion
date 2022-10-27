//
//  File.swift
//  
//
//  Created by Morten Just on 10/27/22.
//

import Foundation
import Combine
import CoreGraphics

public extension Diffusion {
    
    
    /// Generate an image asynchronously. Optional callback with progress and intermediate image. Run inside Task.detached to run in background.
    /// ```
    ///   // without progress reporting
    ///   let image = await diffusion.generate("astronaut in the ocean")
    ///
    ///   // with progress
    ///    let image = await diffusion.generate("astronaut in the ocean") { progress in
    ///           print("progress: \(progress.progress)") }
     func generate(prompt: String,
                         negativePrompt: String = "",
                         seed: Int = Int.random(in: 0...Int.max),
                         steps:Int = 20,
                         guidanceScale:Float = 7.5,
                         progress: ((GenResult) -> Void)? = nil
    ) async -> CGImage? {
        
        return await withCheckedContinuation { continuation in
            self.generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale) { (image, progressFloat, stage) in
                let genResult = GenResult(image: image, progress: Double(progressFloat), stage: stage)
                progress?(genResult)
                if progressFloat == 1, let finalImage = genResult.image {
                    continuation.resume(returning: finalImage)
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
        
    }

    
    /// Generate an image and get a publisher for progress and intermediate image.
    /// ```
    ///     generate("Astronaut in the ocean")
    ///      .sink { result in
    ///         print("progress: \(result.progress)") // result also contains the intermediate image
    ///      }
    ///
     func generate(prompt: String, negativePrompt: String = "", seed: Int = Int.random(in: 0...Int.max), steps:Int = 20, guidanceScale:Float = 7.5) -> AnyPublisher<GenResult,Never> {
        
        let publisher = PassthroughSubject<GenResult, Never>()
        
        Task.detached(priority: .userInitiated) {
            self.generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale) { (cgImage, progress, stage) in
                Task {
                    print("RAW progress", progress)
                    await MainActor.run {
                        let result = GenResult(image: cgImage, progress: Double(progress), stage: stage)
                        publisher.send(result)
                        if progress >= 1 { publisher.send(completion: .finished)}
                    }
                }
            }
        }
        
        return publisher.eraseToAnyPublisher()
        
    }
    
    
    /// Generate and image with a callback.
    ///
    func generate(prompt: String, negativePrompt: String = "", seed: Int = Int.random(in: 0...Int.max), steps:Int = 20, guidanceScale:Float = 7.5,
                         completion: @escaping (CGImage?, Float, String)->()) {
        
        
        mapleDiffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale) { cgImage, progress, stage in
            
            var realProgress = progress
            if progress == 1 && stage.contains("Decoding") {
                realProgress = 0.97
            }
            
            completion(cgImage, realProgress, stage)
            
        }
    }
    
    
}
