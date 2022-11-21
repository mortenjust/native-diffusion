//
//  File.swift
//  
//
//  Created by Morten Just on 10/27/22.
//

import Foundation
import Combine
import CoreGraphics

/**
 Generator functions. Mostly convenience wrappers around generate:input[..]
 
 */

public extension Diffusion {
    

    
    
    /// Generate an image async. Optional callback with progress and intermediate image.
     func generate(
        input: SampleInput,
        progress: ((GenResult) -> Void)? = nil,
        remoteUrl: String? = nil
    ) async -> CGImage? {
         
         var combinedSteps : Double = 1
         var combinedProgress : Double = 0
         
         // maybe load model first
         if modelIsCold,  let remoteUrl, let url = URL(string: remoteUrl) {
             combinedSteps += 1
             print("async gen: cold model, loading")
             try! await prepModels(remoteURL: url) { p in
                 combinedProgress = (p/combinedSteps)
                 let res = GenResult(image: nil, progress: combinedProgress, stage: "Loading Model")
                 progress?(res)
             }
         }
         
         // generate image
        return await withCheckedContinuation { continuation in
            print("async gen: generating", input.prompt)
            self.generate(input: input) { (image, progressFloat, stage) in
                let genResult = GenResult(image: image, progress: Double(progressFloat), stage: stage)
                progress?(genResult)
                
                print("async gen: ", genResult.stage)
                
                if progressFloat == 1, let finalImage = genResult.image {
                    continuation.resume(returning: finalImage)
                }
                
                if progressFloat == 1, genResult.image == nil {
                    continuation.resume(returning: nil)
                }
            }
        }
        
    }
    
    
    /// Generate an image asynchronously. Optional callback with progress and intermediate image. Run inside a Task.detached to run in background.
    /// ```
    ///   // without progress reporting
    ///   let image = await diffusion.generate("astronaut in the ocean")
    ///
    ///   // with progress
    ///    let image = await diffusion.generate("astronaut in the ocean") { progress in
    ///           print("progress: \(progress.progress)") }
    func generate(prompt: String,
                            negativePrompt: String = "",
                  inputImage: CGImage? = nil,
                            seed: Int = Int.random(in: 0...Int.max),
                            steps:Int = 20,
                            guidanceScale:Float = 7.5,
                            progress: ((GenResult) -> Void)? = nil,
                      remoteUrl: String? = nil
    ) async -> CGImage? {
        /// This is a just a wrapper that adds `SampleInput`, mainly for simplicity and to not break pre-SampleInput builds.
        let sampleInput = SampleInput(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale)
        return await generate(input: sampleInput, progress: progress)
    }
    
    

    
    /// Generate an image and get a publisher for progress and intermediate image. Runs detached with "initiated" priority.
    /// ```
    ///     diffusion.generate("Astronaut in the ocean")
    ///      .sink { result in
    ///         print("progress: \(result.progress)") // result also contains the intermediate image
    ///      }.store(in: ...)
    ///
     func generate(input: SampleInput) -> AnyPublisher<GenResult,Never> {
        let publisher = PassthroughSubject<GenResult, Never>()
        
        Task.detached(priority: .userInitiated) { // TODO: Test other priorities and consider making it an optional argument
            self.generate(input: input) { (cgImage, progress, stage) in
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
    
    func generate(prompt: String, negativePrompt: String = "", seed: Int = Int.random(in: 0...Int.max), steps:Int = 20, guidanceScale:Float = 7.5) -> AnyPublisher<GenResult,Never> {
        
        let sampleInput = SampleInput(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale)
        
        return generate(input: sampleInput)
        
    }
    
    
    
    
    
    /// Generate and image with a callback on current thread.
    ///
    func generate(
        input: SampleInput,
        completion: @escaping (CGImage?, Float, String)->()
    ) {
        mapleDiffusion.generate(input: input) { cgImage, progress, stage in
            // penultimate step is also marked progress 1 in MD currently, working around it
            var realProgress = progress
            if progress == 1 && stage.contains("Decoding") {
                realProgress = 0.97
            }
            
            completion(cgImage, realProgress, stage)
        }
    }
}


