import Foundation
import Combine
import CoreGraphics
import CoreImage

/**
 
 This is the package's wrapper for the `MapleDiffusion` class. It adds a few convenient ways to use @madebyollin's code.
 
 */

public class Diffusion : ObservableObject {
    
    /// Current state of the models. Only supports states for loading currently, updated via `initModels`
    public var state = PassthroughSubject<GeneratorState, Never>()
    
    @MainActor
    @Published public var isModelReady = false
    @Published public var loadingProgress : Double = 0.0
    
    var mapleDiffusion : MapleDiffusion!
    
//    public var width : NSNumber { mapleDiffusion.width }
//    public var height : NSNumber { mapleDiffusion.height }
    
    private var saveMemory = false
    
    /// Initializes the main class with no side effects.
    ///
    /// ```
    /// // Use a local folder
    /// let diffusion = Diffusion(modelFolder: URL...))
    ///
    ///// Download a folder
    /// let diffusion = Diffusion(modelZipURL: URL(string: "https://example.com/myfiles.zip")
    ///
    /// // Use directly in SwiftUI
    /// @StateObject var Diffusion [...]
    ///
    /// ```
    ///
    /// - Parameters:
    ///   - saveMemoryButBeSlower: Recommended for iOS or old macs
    ///   - modelZipUrl: Optional. The destination on the web where you are hosting your converted model files. The class will download and extract the model files if the model folder is empty.
    ///   - modelFolder: Optional. The local folder. Application Support is recommended as that will not require special permissions. If you don't provide a value, the class will a folder in Application Support using the bundle identifier.
    ///
    /// - Important: This will not load the models or start downloading. Call `loadModels` before you start generating. Don't forget to convert your model to bins first. See README
    
    
    public init(saveMemoryButBeSlower: Bool = false) {
        self.saveMemory = saveMemoryButBeSlower
        state.send(.notStarted)
    }

    /// Empty publisher for convenience in SwiftUI (since you can't listen to a nil)
    public static var placeholderPublisher : AnyPublisher<GenResult,Never> { PassthroughSubject<GenResult,Never>().eraseToAnyPublisher() }
    

    
    
    /**
     #Async wrappers
     Creating async sequences for MD's callbacks
     */
    
    /// Tuple type for easier access to the progress while also getting the full state.
    public typealias LoaderUpdate = (progress: Double, state: GeneratorState)
    
    /// Init models and update publishers. Run this off the main actor.
    /// TODO: 3 overloads: local only, remote only, both - build into arguments what they do
    var combinedProgress : Double = 0
    
    
    public func prepModels(localUrl: URL, progress:((Double)->Void)? = nil) async throws {
        print("prep: with localurl")
        let fetcher = ModelFetcher(local: localUrl)
        try await initModels(fetcher: fetcher, progress: progress)
    }
    
    public func prepModels(remoteURL: URL, progress:((Double)->Void)? = nil) async throws {
        print("prep: with remote")
        let fetcher = ModelFetcher(remote: remoteURL)
        try await initModels(fetcher: fetcher, progress: progress)
    }
    
    public func prepModels(localUrl: URL, remoteUrl: URL, progress:((Double)->Void)? = nil) async throws {
        print("prep: with remote and local")
        let fetcher = ModelFetcher(local: localUrl, remote: remoteUrl)
        try await initModels(fetcher: fetcher, progress: progress)
    }
    
    private func initModels(
        fetcher:ModelFetcher,
        progress:((Double)->Void)?
    ) async throws {

        // TODO: Don't use 2 steps if the model is downloaded
        let combinedSteps = 2.0 // 1. fetch, 2. init
        var combinedProgress = 0.0
        
        print("prep: maybe downloading")
        
        /// 1. Fetch the model and put it in the global var used by MD
        global_modelFolder = try await fetcher.fetch { p in
            combinedProgress = (p/combinedSteps)
            print("fetcher says", p, "combined", combinedProgress)
            self.updateLoadingProgress(progress: combinedProgress, message: "Fetching models")
        }
        
        /// 2. instantiate MD, which has light side effects
        self.mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: saveMemory)
        
        print("prep: init start")
        
        let earlierProgress = combinedProgress
        
        /// 3. Initialize models on a background thread
        try await initModels() { p in
            combinedProgress = (p/combinedSteps) + earlierProgress
            print("initmodel says", p, "combined", combinedProgress)
            self.updateLoadingProgress(progress: combinedProgress, message: "Loading models")
            progress?(combinedProgress)
        }
        
        /// 4. Done. Set published main status to true
        await MainActor.run {
            print("model is now ready")
            self.state.send(.ready)
            self.loadingProgress = 1
            self.isModelReady = true
        }
        
  
        print("prep: init done")
        
    }
    
    private func updateLoadingProgress(progress: Double, message:String) {
        Task {
            await MainActor.run {
                self.state.send(GeneratorState.modelIsLoading(progress: progress, message: message))
                self.loadingProgress = progress
            }
        }
    }
    

    
    /// Async wrapper for initModels
    private func initModels(progressCompletion: ProgressClosure?) async throws {
        
        return try await withCheckedThrowingContinuation { continuation in
            
            Task.detached {
                self.initModels { progress, stage in
                    print("MD says", progress, stage)
                    progressCompletion?(Double(progress))
                    // this is where we are: This is being called twice, the "ready!"
                    if progress == 1 { continuation.resume(returning: ()) }
                }
            }
        }
    }
    
    
    /**
     # Light wrappers
     Just passing through to MD with the same interface
     */
    
    ///  initialize SD models and send updates through a callback. Private because it assumes the global variable `global_modelFolder` is set..
    ///  TODO: Use ModelFetcher here
    private func initModels(completion: @escaping (Float, String)->()) {
        mapleDiffusion.initModels { progress, stage in
            print("DeepMD says:", progress, stage)
            
            /// Workaround to start diffusion at launch rather than at first image. For the user, this makes the first image start immediately, at the tradeoff of a slower initial boot. We'll just throw it in there, without waiting for it
            if progress == 1 {
                Task.detached {
                    print("Last stage!")
                    self.mapleDiffusion.generate(prompt: "", negativePrompt: "", seed: 42, steps: 1, guidanceScale: 0) { _, progress, _ in
                        print("diffusion started succesfully")
                    }
                }
            }
            
            completion(progress, stage)
        }
    }
    

}
