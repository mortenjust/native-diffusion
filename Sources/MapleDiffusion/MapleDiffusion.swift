
import MetalPerformanceShadersGraph
import Foundation
import AppKit
import Combine


public struct ProgressMonitor : Equatable {
    public var completed: Double = 0
    public var total: Double
    public var message: String = ""
    public var fractionCompleted : Double { completed/total }
    
    mutating func update(completed: Double, message:String) {
        self.completed = completed
        self.message = message
    }
    
    mutating func with(completed:Double, message:String) -> ProgressMonitor {
        self.completed = completed
        self.message = message
        return self
    }
    
    mutating func increasing(withMessage message:String) -> ProgressMonitor {
        self.completed += 1
        self.message = message
        return self
    }
    
    public static func example(_ progress : Double) -> ProgressMonitor {
        ProgressMonitor(completed: progress, total: 1, message: "Loading files")
    }
    
}

@available(macOS 12.3, *)
public class MapleDiffusion : ObservableObject {
    @Published public var isModelLoaded = false
    
    public enum GeneratorState: Equatable {
        public static func == (lhs: MapleDiffusion.GeneratorState, rhs: MapleDiffusion.GeneratorState) -> Bool {
            switch(lhs, rhs) {
            case (.ready, .ready):
                return true
            case (.modelIsLoading( _), .modelIsLoading( _)):
                return true
            case (.notStarted, .notStarted):
                return true
            default:
                return false
            }
        }
        
        case notStarted
        case modelIsLoading(progress: ProgressMonitor)
        case ready
    }
    
    
    public var state = PassthroughSubject<GeneratorState, Never>()
    
    enum ModelLoadError : Error {
        case whileLoading
        case other
    }
    
    enum GenerationError : Error {
        case placeholder(String)
    }
    
    var device: MTLDevice?
    var graphDevice: MPSGraphDevice?
    var commandQueue: MTLCommandQueue?
    var saveMemory: Bool?
    
    // text tokenization
    var tokenizer: BPETokenizer?
    
    // text guidance
    var textGuidanceExecutable: MPSGraphExecutable??
    
    // time embedding
    var tembGraph: MPSGraph?
    var tembTIn: MPSGraphTensor?
    var tembOut: MPSGraphTensor?
    
    // diffusion
    var diffGraph: MPSGraph?
    var diffGuidanceScaleIn: MPSGraphTensor?
    var diffXIn: MPSGraphTensor?
    var diffEtaUncondIn: MPSGraphTensor?
    var diffEtaCondIn: MPSGraphTensor?
    var diffTIn: MPSGraphTensor?
    var diffTPrevIn: MPSGraphTensor?
    var diffOut: MPSGraphTensor?
    var diffAuxOut: MPSGraphTensor?
    
    // unet
    // MEM-HACK: split into subgraphs
    var unetAnUnexpectedJourneyExecutable: MPSGraphExecutable?
    var anUnexpectedJourneyShapes = [[NSNumber]]()
    var unetTheDesolationOfSmaugExecutable: MPSGraphExecutable?
    var theDesolationOfSmaugShapes = [[NSNumber]]()
    var theDesolationOfSmaugIndices = [MPSGraphTensor: Int]()
    var unetTheBattleOfTheFiveArmiesExecutable: MPSGraphExecutable?
    var theBattleOfTheFiveArmiesIndices = [MPSGraphTensor: Int]()
    
    public var width: NSNumber = 64
    public var height: NSNumber = 64
    
    
    public init(saveMemoryButBeSlower: Bool = true,
                modelFolder mf : URL? = Bundle.main.url(forResource: "bins", withExtension: nil)) {
        // set global folder
        modelFolder = mf
//        try? loadModel(saveMemoryButBeSlower: saveMemoryButBeSlower)
    }
    
    func updateState(_ newState:GeneratorState) {
        Task {
            await MainActor.run(body: {
                self.state.send(newState)
            })
        }
    }
    
    /// Loads the model in a Task, but can still be heavy, probably due to the MPS stuff
    public func loadModel(saveMemoryButBeSlower: Bool = true) throws {
        
        Task {
            var progress = ProgressMonitor(total: 11)
            
            
            self.updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Creating device and queue")))
            
            print("instantiate device and queue")
            saveMemory = saveMemoryButBeSlower
            guard let device = MTLCreateSystemDefaultDevice() else { throw ModelLoadError.whileLoading }
            graphDevice = MPSGraphDevice(mtlDevice: device)
            commandQueue = device.makeCommandQueue()!
            
            // text tokenization
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Creating tokenizer")))
            print("instantiate tokenizer")
            tokenizer = BPETokenizer()
            
            // time embedding
            print("make temb graph")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Making temporal graph")))
            tembGraph = makeGraph()
            tembTIn = tembGraph!.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
            tembOut = makeTimeFeatures(graph: tembGraph!, tIn: tembTIn!) // force
            
            // diffusion
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Creating diffusion graph")))
            print("make diffusion graph")
            diffGraph = makeGraph()
            guard let diffGraph else {
                print("bailing")
                throw ModelLoadError.whileLoading }
            
            print("setup placeholders")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Setting up placeholders")))
            
            // these could probably be wrapped in their own struct to simplify
            diffXIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
            diffEtaUncondIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
            diffEtaCondIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
            diffTIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
            diffTPrevIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
            diffGuidanceScaleIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.float16, name: nil)
            
            print("set up diffusion")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Setting up diffusion")))
            
            diffOut = makeDiffusionStep(graph: diffGraph, xIn: diffXIn!, etaUncondIn: diffEtaUncondIn!, etaCondIn: diffEtaCondIn!, tIn: diffTIn!, tPrevIn: diffTPrevIn!, guidanceScaleIn: diffGuidanceScaleIn!)
            diffAuxOut = makeAuxUpsampler(graph: diffGraph, xIn: diffOut!)
            
            
            print("instantiate text guidance")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Initializing text guidance")))
            // text guidance
            initTextGuidance()
            
            // unet
            print("instantiate unet 1")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Initializing U-Net 1")))
            try initAnUnexpectedJourney()
            print("instantiate unet 2")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Initializing U-Net 2")))
            try initTheDesolationOfSmaug()
            print("instantiate unet 3")
            updateState(.modelIsLoading(progress: progress.increasing(withMessage: "Initializing U-Net 3")))
            try initTheBattleOfTheFiveArmies()
            
             self.updateState(.ready)
            print("model loading done")
            DispatchQueue.main.async {
                self.isModelLoaded = true
            }
            
        }
    }
    
    private func initTextGuidance() {
        let graph = makeGraph()
        let textGuidanceIn = graph.placeholder(shape: [2, 77], dataType: MPSDataType.int32, name: nil)
        let textGuidanceOut = makeTextGuidance(graph: graph, xIn: textGuidanceIn, name: "cond_stage_model.transformer.text_model")
        let textGuidanceOut0 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 0, length: 1, name: nil)
        let textGuidanceOut1 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 1, length: 1, name: nil)
        textGuidanceExecutable = graph.compile(with: graphDevice, feeds: [textGuidanceIn: MPSGraphShapedType(shape: textGuidanceIn.shape, dataType: MPSDataType.int32)], targetTensors: [textGuidanceOut0, textGuidanceOut1], targetOperations: nil, compilationDescriptor: nil)
    }
    
    private func initAnUnexpectedJourney() throws {
        guard let saveMemory else { throw ModelLoadError.other }
        
        let graph = makeGraph()
        let xIn = graph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let tembIn = graph.placeholder(shape: [1, 320], dataType: MPSDataType.float16, name: nil)
        let unetOuts = makeUNetAnUnexpectedJourney(graph: graph, xIn: xIn, tembIn: tembIn, condIn: condIn, name: "model.diffusion_model", saveMemory: saveMemory)
        let unetFeeds = [xIn, condIn, tembIn].reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        unetAnUnexpectedJourneyExecutable = graph.compile(with: graphDevice, feeds: unetFeeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        anUnexpectedJourneyShapes = unetOuts.map{$0.shape!}
    }
    
    private func initTheDesolationOfSmaug() throws {
        guard let saveMemory else { throw ModelLoadError.other }
        
        let graph = makeGraph()
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let placeholders = anUnexpectedJourneyShapes.map{graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)} + [condIn]
        theDesolationOfSmaugIndices.removeAll()
        for i in 0..<placeholders.count {
            theDesolationOfSmaugIndices[placeholders[i]] = i
        }
        let feeds = placeholders.reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        let unetOuts = makeUNetTheDesolationOfSmaug(graph: graph, savedInputsIn: placeholders, name: "model.diffusion_model", saveMemory: saveMemory)
        unetTheDesolationOfSmaugExecutable = graph.compile(with: graphDevice, feeds: feeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        theDesolationOfSmaugShapes = unetOuts.map{$0.shape!}
    }
    
    private func initTheBattleOfTheFiveArmies() throws {
        guard let saveMemory else { throw ModelLoadError.other }
        
        let graph = makeGraph()
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let unetPlaceholders = theDesolationOfSmaugShapes.map{graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)} + [condIn]
        theBattleOfTheFiveArmiesIndices.removeAll()
        for i in 0..<unetPlaceholders.count {
            theBattleOfTheFiveArmiesIndices[unetPlaceholders[i]] = i
        }
        let feeds = unetPlaceholders.reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        let unetOut = makeUNetTheBattleOfTheFiveArmies(graph: graph, savedInputsIn: unetPlaceholders, name: "model.diffusion_model", saveMemory: saveMemory)
        unetTheBattleOfTheFiveArmiesExecutable = graph.compile(with: graphDevice, feeds: feeds, targetTensors: [unetOut], targetOperations: nil, compilationDescriptor: nil)
    }
    
    private func randomLatent(seed: Int) -> MPSGraphTensorData {
        let graph = makeGraph()
        let out = graph.randomTensor(withShape: [1, height, width, 4], descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!, seed: seed, name: nil)
        return graph.run(feeds: [:], targetTensors: [out], targetOperations: nil)[out]!
    }
    
    private func runTextGuidance(baseTokens: [Int], tokens: [Int]) throws -> (MPSGraphTensorData, MPSGraphTensorData)  {
        
        guard let textGuidanceExecutable, let commandQueue, let graphDevice else { throw GenerationError.placeholder(#function) }
        
        let tokensData = (baseTokens + tokens).map({Int32($0)}).withUnsafeBufferPointer {Data(buffer: $0)}
        let tokensMPSData = MPSGraphTensorData(device: graphDevice, data: tokensData, shape: [2, 77], dataType: MPSDataType.int32)
        let res = textGuidanceExecutable!.run(with: commandQueue, inputs: [tokensMPSData], results: nil, executionDescriptor: nil)
        return (res[0], res[1])
    }
    
    private func loadDecoderAndGetFinalImage(xIn: MPSGraphTensorData) throws -> MPSGraphTensorData {
        // MEM-HACK: decoder is loaded from disc and deallocated to save memory (at cost of latency)
        
        guard let commandQueue else { throw GenerationError.placeholder(#function) }
        
        let x = xIn
        let decoderGraph = makeGraph()
        let decoderIn = decoderGraph.placeholder(shape: x.shape, dataType: MPSDataType.float16, name: nil)
        let decoderOut = makeDecoder(graph: decoderGraph, xIn: decoderIn)
        return decoderGraph.run(with: commandQueue, feeds: [decoderIn: x], targetTensors: [decoderOut], targetOperations: nil)[decoderOut]!
    }
    
    private func reorderAnUnexpectedJourney(x: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        var out = [MPSGraphTensorData]()
        for r in unetAnUnexpectedJourneyExecutable!.feedTensors! {
            for i in x {
                if (i.shape == r.shape) {
                    out.append(i)
                }
            }
        }
        return out
    }
    
    private func reorderTheDesolationOfSmaug(x: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        var out = [MPSGraphTensorData]()
        for r in unetTheDesolationOfSmaugExecutable!.feedTensors! {
            out.append(x[theDesolationOfSmaugIndices[r]!])
        }
        return out
    }
    
    private func reorderTheBattleOfTheFiveArmies(x: [MPSGraphTensorData]) -> [MPSGraphTensorData] {
        var out = [MPSGraphTensorData]()
        for r in unetTheBattleOfTheFiveArmiesExecutable!.feedTensors! {
            out.append(x[theBattleOfTheFiveArmiesIndices[r]!])
        }
        return out
    }
    
    private func runUNet(latent: MPSGraphTensorData, guidance: MPSGraphTensorData, temb: MPSGraphTensorData) throws -> MPSGraphTensorData {
        guard let commandQueue else { throw GenerationError.placeholder(#function) }
        
        var x = unetAnUnexpectedJourneyExecutable!.run(with: commandQueue, inputs: reorderAnUnexpectedJourney(x: [latent, guidance, temb]), results: nil, executionDescriptor: nil)
        x = unetTheDesolationOfSmaugExecutable!.run(with: commandQueue, inputs: reorderTheDesolationOfSmaug(x: x + [guidance]), results: nil, executionDescriptor: nil)
        return unetTheBattleOfTheFiveArmiesExecutable!.run(with: commandQueue, inputs: reorderTheBattleOfTheFiveArmies(x: x + [guidance]), results: nil, executionDescriptor: nil)[0]
    }
    
    private func runBatchedUNet(latent: MPSGraphTensorData, baseGuidance: MPSGraphTensorData, textGuidance: MPSGraphTensorData, temb: MPSGraphTensorData) throws -> (MPSGraphTensorData, MPSGraphTensorData) {
        // concat
        var graph = makeGraph()
        let bg = graph.placeholder(shape: baseGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let tg = graph.placeholder(shape: textGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let concatGuidance = graph.concatTensors([bg, tg], dimension: 0, name: nil)
        let concatGuidanceData = graph.run(feeds: [bg : baseGuidance, tg: textGuidance], targetTensors: [concatGuidance], targetOperations: nil)[concatGuidance]!
        // run
        let concatEtaData = try runUNet(latent: latent, guidance: concatGuidanceData, temb: temb)
        // split
        graph = makeGraph()
        let etas = graph.placeholder(shape: concatEtaData.shape, dataType: concatEtaData.dataType, name: nil)
        let eta0 = graph.sliceTensor(etas, dimension: 0, start: 0, length: 1, name: nil)
        let eta1 = graph.sliceTensor(etas, dimension: 0, start: 1, length: 1, name: nil)
        let etaRes = graph.run(feeds: [etas: concatEtaData], targetTensors: [eta0, eta1], targetOperations: nil)
        return (etaRes[eta0]!, etaRes[eta1]!)
    }
    
    private func generateLatent(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String) -> ()) throws -> MPSGraphTensorData {
        
        completion(nil, 0, "Tokenizing...")
        
        // 1. String -> Tokens
        
        let baseTokens = tokenizer!.encode(s: negativePrompt)
        let tokens = tokenizer!.encode(s: prompt)
        completion(nil, 0.25 * 1 / Float(steps), "Encoding...")
        
        // 2. Tokens -> Embedding
        let (baseGuidance, textGuidance) = try runTextGuidance(baseTokens: baseTokens, tokens: tokens)
        if (saveMemory!) {
            // MEM-HACK unload the text guidance to fit the unet
            textGuidanceExecutable = nil
        }
        completion(nil, 0.5 * 1 / Float(steps), "Generating noise...")
        
        // 3. Noise generation
        var latent = randomLatent(seed: seed)
        
        let timesteps = Array<Int>(stride(from: 1, to: 1000, by: Int(1000 / steps)))
        completion(nil, 0.75 * 1 / Float(steps), "Starting diffusion...")
        
        // 4. Diffusion
        for t in (0..<timesteps.count).reversed() {
            let tick = CFAbsoluteTimeGetCurrent()
            
            // step
            let tsPrev = t > 0 ? timesteps[t - 1] : timesteps[t] - 1000 / steps
            let tData = [Int32(timesteps[t])].withUnsafeBufferPointer {Data(buffer: $0)}
            let tMPSData = MPSGraphTensorData(device: graphDevice!, data: tData, shape: [1], dataType: MPSDataType.int32)
            let tPrevData = [Int32(tsPrev)].withUnsafeBufferPointer {Data(buffer: $0)}
            let tPrevMPSData = MPSGraphTensorData(device: graphDevice!, data: tPrevData, shape: [1], dataType: MPSDataType.int32)
            let guidanceScaleData = [Float32(guidanceScale)].withUnsafeBufferPointer {Data(buffer: $0)}

            let guidanceScaleMPSData = MPSGraphTensorData(device: graphDevice!, data: guidanceScaleData, shape: [1], dataType: MPSDataType.float16)
            let temb = tembGraph!.run(with: commandQueue!, feeds: [tembTIn!: tMPSData], targetTensors: [tembOut!], targetOperations: nil)[tembOut!]!
            let etaUncond: MPSGraphTensorData
            let etaCond: MPSGraphTensorData
            if (saveMemory!) {
                // MEM-HACK: un/neg-conditional and text-conditional are run in two separate passes (not batched) to save memory
                etaUncond = try runUNet(latent: latent, guidance: baseGuidance, temb: temb)
                etaCond = try runUNet(latent: latent, guidance: textGuidance, temb: temb)
            } else {
                (etaUncond, etaCond) = try runBatchedUNet(latent: latent, baseGuidance: baseGuidance, textGuidance: textGuidance, temb: temb)
            }
            
            let res = diffGraph!.run(
                with: commandQueue!,
                feeds: [diffXIn!: latent,
                diffEtaUncondIn!: etaUncond,
                  diffEtaCondIn!: etaCond,
                        diffTIn!: tMPSData,
                    diffTPrevIn!: tPrevMPSData,
            diffGuidanceScaleIn!: guidanceScaleMPSData],
                targetTensors: [diffOut!, diffAuxOut!],
                targetOperations: nil)
            latent = res[diffOut!]!
            
            // update ui
            let tock = CFAbsoluteTimeGetCurrent()
            let stepRuntime = String(format:"%.2fs", tock - tick)
            let progressDesc = t == 0 ? "Decoding..." : "Step \(timesteps.count - t) / \(timesteps.count) (\(stepRuntime) / step)"
            
            let progress = Float(timesteps.count - t) / Float(timesteps.count)
            completion(tensorToCGImage(data: res[diffAuxOut!]!),
                       progress - 0.01, // 0.01 is buffer for last step
                       progressDesc)
        }
        return latent
    }
    
    public struct GenResult {
        internal init(image: CGImage?, progress: Double, stage: String) {
            self.image = image
            self.progress = progress
            self.stage = stage
        }
        
        public let image : CGImage?
        public let progress : Double
        public let stage : String
    }
    
    // MARK: Generate with publisher
    public func generate(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float) throws
    -> AnyPublisher<GenResult, Never>
    {
        let subject = PassthroughSubject<GenResult, Never>()
        
        Task {
            try generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale) { (cgImage, progress, stage) in
                
                let result = GenResult(image: cgImage, // TODO: Unsafe
                                       progress: Double(progress),
                                       stage: stage)
                subject.send(result)
                if(progress >= 1) {
                    subject.send(completion: .finished)
                }
            }
        }
        return subject.eraseToAnyPublisher()
    }
    
    public func generate(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String)->()) throws {
        
        guard let saveMemory else { throw GenerationError.placeholder(#function) }
        
        let latent = try generateLatent(prompt: prompt,
                                    negativePrompt: negativePrompt,
                                    seed: seed,
                                    steps: steps,
                                    guidanceScale: guidanceScale,
                                    completion: completion)
        
        if (saveMemory) {
            // MEM-HACK: unload the unet to fit the decoder
            unetAnUnexpectedJourneyExecutable = nil
            unetTheDesolationOfSmaugExecutable = nil
            unetTheBattleOfTheFiveArmiesExecutable = nil
        }
        
        // 5. Decoder
        let decoderRes = try loadDecoderAndGetFinalImage(xIn: latent)
        completion(tensorToCGImage(data: decoderRes), 1.0, "Cooling down...")
        
        if (saveMemory) {
            // reload the unet and text guidance
            try initAnUnexpectedJourney()
            try initTheDesolationOfSmaug()
            try initTheBattleOfTheFiveArmies()
            initTextGuidance()
        }
    }
}



