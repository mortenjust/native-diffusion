import MetalPerformanceShadersGraph
import Foundation

// Maple Diffusion implements stable diffusion (original v1.4 model)
// inference via MPSGraph. iOS has a hard memory limit of 4GB (with
// a special entitlement), so this implementation trades off latency
// for memory usage in many places (tagged with MEM-HACK) in order to
// stay under the limit and minimize probability of oom.

/**
 
 When updating
 1. Paste in the entire new file
 2. Replace references to Bundle.main like this
 
 `let fileUrl: URL = Bundle.main.url(forResource: "bins/" + name + (fp32 ? "_fp32" : ""), withExtension: ".bin")!`
 to
`let fileUrl: URL = modelFolder.appendingPathComponent(name + (fp32 ? "_fp32" : "")).appendingPathExtension("bin")`
 
 and also
 ```
 let vocabFile = try! String(
     contentsOf: modelFolder
         .appendingPathComponent("bpe_simple_vocab_16e6")
         .appendingPathExtension("txt")
 )
 ```
 
 */

// madebyollin's code starts here:

class MapleDiffusion {
    let device: MTLDevice
    let graphDevice: MPSGraphDevice
    let commandQueue: MTLCommandQueue
    let saveMemory: Bool
    let shouldSynchronize: Bool
    let modelLocation: URL
    
    // text guidance
    private var _textGuidance: TextGuidance?
    var textGuidance: TextGuidance {
        if let _textGuidance {
            return _textGuidance
        }
        let textGuidance = TextGuidance(
            synchronize: shouldSynchronize,
            device: graphDevice,
            modelLocation: modelLocation
        )
        _textGuidance = textGuidance
        return textGuidance
    }
    
    lazy var diffuser: Diffuser = {
        Diffuser(
            synchronize: shouldSynchronize,
            device: graphDevice,
            modelLocation: modelLocation,
            shape: [1, height, width, 4]
        )
    }()
    
    // unet
    // MEM-HACK: split into subgraphs
    var unetAnUnexpectedJourneyExecutable: MPSGraphExecutable?
    var anUnexpectedJourneyShapes = [[NSNumber]]()
    var unetTheDesolationOfSmaugExecutable: MPSGraphExecutable?
    var theDesolationOfSmaugShapes = [[NSNumber]]()
    var theDesolationOfSmaugIndices = [MPSGraphTensor: Int]()
    var unetTheBattleOfTheFiveArmiesExecutable: MPSGraphExecutable?
    var theBattleOfTheFiveArmiesIndices = [MPSGraphTensor: Int]()
    
    var width: NSNumber = 64
    var height: NSNumber = 64
    
    public init(modelLocation: URL, saveMemoryButBeSlower: Bool = true) {
        self.modelLocation = modelLocation
        saveMemory = saveMemoryButBeSlower
        device = MTLCreateSystemDefaultDevice()!
        graphDevice = MPSGraphDevice(mtlDevice: device)
        commandQueue = device.makeCommandQueue()!
        shouldSynchronize = !device.hasUnifiedMemory
    }
    
    public func initModels(completion: (Float, String)->()) {
        // unet
        completion(0.25, "Loading UNet part 1/3...")
        initAnUnexpectedJourney()
        completion(0.5, "Loading UNet part 2/3...")
        initTheDesolationOfSmaug()
        completion(0.75, "Loading UNet part 3/3...")
        initTheBattleOfTheFiveArmies()
        completion(1, "Loaded models")
    }
    
    private func initAnUnexpectedJourney() {
        let graph = MPSGraph(synchronize: shouldSynchronize)
        let xIn = graph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let tembIn = graph.placeholder(shape: [1, 320], dataType: MPSDataType.float16, name: nil)
        let unetOuts = graph.makeUNetAnUnexpectedJourney(at: modelLocation, xIn: xIn, tembIn: tembIn, condIn: condIn, name: "model.diffusion_model", saveMemory: saveMemory)
        let unetFeeds = [xIn, condIn, tembIn].reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        unetAnUnexpectedJourneyExecutable = graph.compile(with: graphDevice, feeds: unetFeeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        anUnexpectedJourneyShapes = unetOuts.map{$0.shape!}
    }
    
    private func initTheDesolationOfSmaug() {
        let graph = MPSGraph(synchronize: shouldSynchronize)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let placeholders = anUnexpectedJourneyShapes.map{graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)} + [condIn]
        theDesolationOfSmaugIndices.removeAll()
        for i in 0..<placeholders.count {
            theDesolationOfSmaugIndices[placeholders[i]] = i
        }
        let feeds = placeholders.reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        let unetOuts = graph.makeUNetTheDesolationOfSmaug(at: modelLocation, savedInputsIn: placeholders, name: "model.diffusion_model", saveMemory: saveMemory)
        unetTheDesolationOfSmaugExecutable = graph.compile(with: graphDevice, feeds: feeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        theDesolationOfSmaugShapes = unetOuts.map{$0.shape!}
    }
    
    private func initTheBattleOfTheFiveArmies() {
        let graph = MPSGraph(synchronize: shouldSynchronize)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let unetPlaceholders = theDesolationOfSmaugShapes.map{graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)} + [condIn]
        theBattleOfTheFiveArmiesIndices.removeAll()
        for i in 0..<unetPlaceholders.count {
            theBattleOfTheFiveArmiesIndices[unetPlaceholders[i]] = i
        }
        let feeds = unetPlaceholders.reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        let unetOut = graph.makeUNetTheBattleOfTheFiveArmies(at: modelLocation, savedInputsIn: unetPlaceholders, name: "model.diffusion_model", saveMemory: saveMemory)
        unetTheBattleOfTheFiveArmiesExecutable = graph.compile(with: graphDevice, feeds: feeds, targetTensors: [unetOut], targetOperations: nil, compilationDescriptor: nil)
    }
    
    private func randomLatent(seed: Int) -> MPSGraphTensorData {
        let graph = MPSGraph(synchronize: shouldSynchronize)
        let out = graph.randomTensor(withShape: [1, height, width, 4], descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!, seed: seed, name: nil)
        return graph.run(with: commandQueue, feeds: [:], targetTensors: [out], targetOperations: nil)[out]!
    }
    
    private func runTextGuidance(prompt: String, negativePrompt: String) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let guidance = textGuidance.run(with: commandQueue, prompt: prompt, negativePrompt: negativePrompt)
        if saveMemory {
            // MEM-HACK unload the text guidance to fit the unet
            _textGuidance = nil
        }
        return guidance
    }
    
    private func loadDecoderAndGetFinalImage(xIn: MPSGraphTensorData) -> MPSGraphTensorData {
        // MEM-HACK: decoder is loaded from disc and deallocated to save memory (at cost of latency)
        let x = xIn
        let decoderGraph = MPSGraph(synchronize: shouldSynchronize)
        let decoderIn = decoderGraph.placeholder(shape: x.shape, dataType: MPSDataType.float16, name: nil)
        let decoderOut = decoderGraph.makeDecoder(at: modelLocation, xIn: decoderIn)
        return decoderGraph.run(with: commandQueue, feeds: [decoderIn: x], targetTensors: [decoderOut], targetOperations: nil)[decoderOut]!
    }
    
    private func initLatent(image: CGImage, tEnc: Int, timesteps: MPSGraphTensorData, seed: Int) -> MPSGraphTensorData {
        // MEM-HACK: encoder is loaded from disc and deallocated to save memory (at cost of latency)
        
        let imageData = MPSGraphTensorData(device: graphDevice, cgImage: image)
        
        let graph = MPSGraph(synchronize: shouldSynchronize)
        let encoderIn = graph.placeholder(shape: imageData.shape, dataType: MPSDataType.uInt8, name: nil)
        let encoderOut = graph.makeEncoder(at: modelLocation, xIn: encoderIn)
        let gaussianNoise = graph.randomTensor(withShape: [1, height, width, 4], descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!, seed: seed, name: nil)
        let gaussianOut = graph.diagonalGaussianDistribution(encoderOut, noise: gaussianNoise)
        let scaled = graph.multiplication(gaussianOut, graph.constant(0.18215, dataType: MPSDataType.float16), name: "rescale")
        
        let stepData = tEnc.tensorData(device: graphDevice)
        let stepIn = graph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        let timestepsIn = graph.placeholder(shape: timesteps.shape, dataType: MPSDataType.int32, name: nil)
        
        let noise = graph.randomTensor(withShape: [1, height, width, 4], descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!, seed: seed, name: nil)
        let stochasticEncode = graph.stochasticEncode(at: modelLocation, stepIn: stepIn, timestepsIn: timestepsIn, imageIn: scaled, noiseIn: noise)
        
        return graph.run(
            with: commandQueue,
            feeds: [
                encoderIn: imageData,
                stepIn: stepData,
                timestepsIn: timesteps
            ], targetTensors: [
                gaussianNoise, noise, encoderOut, gaussianOut, scaled, stochasticEncode
            ], targetOperations: nil
        )[stochasticEncode]!
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
    
    private func runUNet(latent: MPSGraphTensorData, guidance: MPSGraphTensorData, temb: MPSGraphTensorData) -> MPSGraphTensorData {
        var x = unetAnUnexpectedJourneyExecutable!.run(with: commandQueue, inputs: reorderAnUnexpectedJourney(x: [latent, guidance, temb]), results: nil, executionDescriptor: nil)
        x = unetTheDesolationOfSmaugExecutable!.run(with: commandQueue, inputs: reorderTheDesolationOfSmaug(x: x + [guidance]), results: nil, executionDescriptor: nil)
        return unetTheBattleOfTheFiveArmiesExecutable!.run(with: commandQueue, inputs: reorderTheBattleOfTheFiveArmies(x: x + [guidance]), results: nil, executionDescriptor: nil)[0]
    }
    
    private func runBatchedUNet(latent: MPSGraphTensorData, baseGuidance: MPSGraphTensorData, textGuidance: MPSGraphTensorData, temb: MPSGraphTensorData) -> (MPSGraphTensorData, MPSGraphTensorData) {
        // concat
        var graph = MPSGraph(synchronize: shouldSynchronize)
        let bg = graph.placeholder(shape: baseGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let tg = graph.placeholder(shape: textGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let concatGuidance = graph.concatTensors([bg, tg], dimension: 0, name: nil)
        let concatGuidanceData = graph.run(with: commandQueue, feeds: [bg : baseGuidance, tg: textGuidance], targetTensors: [concatGuidance], targetOperations: nil)[concatGuidance]!
        // run
        let concatEtaData = runUNet(latent: latent, guidance: concatGuidanceData, temb: temb)
        // split
        graph = MPSGraph(synchronize: shouldSynchronize)
        let etas = graph.placeholder(shape: concatEtaData.shape, dataType: concatEtaData.dataType, name: nil)
        let eta0 = graph.sliceTensor(etas, dimension: 0, start: 0, length: 1, name: nil)
        let eta1 = graph.sliceTensor(etas, dimension: 0, start: 1, length: 1, name: nil)
        let etaRes = graph.run(with: commandQueue, feeds: [etas: concatEtaData], targetTensors: [eta0, eta1], targetOperations: nil)
        return (etaRes[eta0]!, etaRes[eta1]!)
    }
    
    private func generateLatent(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String)->()) -> MPSGraphTensorData {
        completion(nil, 0, "Tokenizing...")
        
        // 1. String -> Embedding
        let guidanceScaleData = [Float32(guidanceScale)].withUnsafeBufferPointer {Data(buffer: $0)}
        let guidanceScaleMPSData = MPSGraphTensorData(device: graphDevice, data: guidanceScaleData, shape: [1], dataType: MPSDataType.float32)
        
        let (baseGuidance, textGuidance) = runTextGuidance(prompt: prompt, negativePrompt: negativePrompt)
        completion(nil, 0.5 * 1 / Float(steps), "Generating noise...")
        
        // 3. Noise generation
        let scheduler = Scheduler(synchronize: shouldSynchronize, device: graphDevice, modelLocation: modelLocation, steps: steps)
        var latent = randomLatent(seed: seed)
        completion(nil, 0.75 * 1 / Float(steps), "Starting diffusion...")
        
        // 4. Diffusion
        for (index, timestep) in scheduler.timesteps.enumerated().reversed() {
            let tick = CFAbsoluteTimeGetCurrent()
            print("Step \(index):", timestep)
            
            // step
            let temb = scheduler.run(with: commandQueue, timestep: timestep)
            
            let etaUncond: MPSGraphTensorData
            let etaCond: MPSGraphTensorData
            if (saveMemory) {
                // MEM-HACK: un/neg-conditional and text-conditional are run in two separate passes (not batched) to save memory
                etaUncond = runUNet(latent: latent, guidance: baseGuidance, temb: temb)
                etaCond = runUNet(latent: latent, guidance: textGuidance, temb: temb)
            } else {
                (etaUncond, etaCond) = runBatchedUNet(latent: latent, baseGuidance: baseGuidance, textGuidance: textGuidance, temb: temb)
            }
            let (newLatent, auxOut) = diffuser.run(
                with: commandQueue,
                latent: latent,
                timestep: timestep,
                timestepSize: scheduler.timestepSize,
                etaUncond: etaUncond,
                etaCond: etaCond,
                guidanceScale: guidanceScaleMPSData
            )
            latent = newLatent
            
            // update ui
            let tock = CFAbsoluteTimeGetCurrent()
            let stepRuntime = String(format:"%.2fs", tock - tick)
            let progressDesc = index == 0 ? "Decoding..." : "Step \(scheduler.timesteps.count - index) / \(scheduler.timesteps.count) (\(stepRuntime) / step)"
            let outImage = auxOut?.cgImage
            completion(outImage, Float(scheduler.timesteps.count - index) / Float(scheduler.timesteps.count), progressDesc)
        }
        return latent
    }
    
    public func generate(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String)->()) {
        let latent = generateLatent(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, guidanceScale: guidanceScale, completion: completion)
        
        if (saveMemory) {
            // MEM-HACK: unload the unet to fit the decoder
            unetAnUnexpectedJourneyExecutable = nil
            unetTheDesolationOfSmaugExecutable = nil
            unetTheBattleOfTheFiveArmiesExecutable = nil
        }
        
        // 5. Decoder
        let decoderRes = loadDecoderAndGetFinalImage(xIn: latent)
        completion(decoderRes.cgImage, 1.0, "Cooling down...")
        
        if (saveMemory) {
            // reload the unet and text guidance
            initAnUnexpectedJourney()
            initTheDesolationOfSmaug()
            initTheBattleOfTheFiveArmies()
        }
    }
    
    private func generateLatent(
        initImage: CGImage,
        prompt: String,
        negativePrompt: String,
        seed: Int,
        steps: Int,
        strength: Float,
        guidanceScale: Float,
        completion: @escaping (CGImage?, Float, String)->()
    ) -> MPSGraphTensorData {
        assert(strength >= 0 && strength <= 1, "Invalid strength value \(strength)")
        completion(initImage, 0, "Tokenizing...")
        
        // 1. String -> Embedding
        let guidanceScaleData = [Float32(guidanceScale)].withUnsafeBufferPointer {Data(buffer: $0)}
        let guidanceScaleMPSData = MPSGraphTensorData(device: graphDevice, data: guidanceScaleData, shape: [1], dataType: MPSDataType.float32)
        
        let (baseGuidance, textGuidance) = runTextGuidance(prompt: prompt, negativePrompt: negativePrompt)
        completion(initImage, 0.5 * 1 / Float(steps), "Generating noise...")
        
        // 2. Noise generation
        let scheduler = Scheduler(synchronize: shouldSynchronize, device: graphDevice, modelLocation: modelLocation, steps: steps)
        let tEnc = Int(Float(steps) * strength)
        var latent = initLatent(image: initImage, tEnc: tEnc, timesteps: scheduler.timestepsData, seed: seed)
        
        completion(nil, 0.75 * 1 / Float(steps), "Starting diffusion...")
        
        // 3. Diffusion
        for (index, timestep) in scheduler.timesteps[0..<tEnc].enumerated().reversed() {
            let tick = CFAbsoluteTimeGetCurrent()
            
            // step
            let temb = scheduler.run(with: commandQueue, timestep: timestep)
            
            let etaUncond: MPSGraphTensorData
            let etaCond: MPSGraphTensorData
            if (saveMemory) {
                // MEM-HACK: un/neg-conditional and text-conditional are run in two separate passes (not batched) to save memory
                etaUncond = runUNet(latent: latent, guidance: baseGuidance, temb: temb)
                etaCond = runUNet(latent: latent, guidance: textGuidance, temb: temb)
            } else {
                (etaUncond, etaCond) = runBatchedUNet(latent: latent, baseGuidance: baseGuidance, textGuidance: textGuidance, temb: temb)
            }
            let (newLatent, auxOut) = diffuser.run(
                with: commandQueue,
                latent: latent,
                timestep: timestep,
                timestepSize: scheduler.timestepSize,
                etaUncond: etaUncond,
                etaCond: etaCond,
                guidanceScale: guidanceScaleMPSData
            )
            latent = newLatent
            
            // update ui
            let tock = CFAbsoluteTimeGetCurrent()
            let stepRuntime = String(format:"%.2fs", tock - tick)
            let progressDesc = index == 0 ? "Decoding..." : "Step \(scheduler.timesteps.count - index) / \(scheduler.timesteps.count) (\(stepRuntime) / step)"
            let outImage = auxOut?.cgImage
            completion(outImage, Float(scheduler.timesteps.count - index) / Float(scheduler.timesteps.count), progressDesc)
        }
        return latent
    }
    
    public func generate(
        initImage: CGImage,
        prompt: String,
        negativePrompt: String,
        seed: Int,
        steps: Int,
        strength: Float,
        guidanceScale: Float,
        completion: @escaping (CGImage?, Float, String)->()
    ) {
        let tick = CFAbsoluteTimeGetCurrent()
        let latent = generateLatent(initImage: initImage, prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: steps, strength: strength, guidanceScale: guidanceScale, completion: completion)
        
        if (saveMemory) {
            // MEM-HACK: unload the unet to fit the decoder
            unetAnUnexpectedJourneyExecutable = nil
            unetTheDesolationOfSmaugExecutable = nil
            unetTheBattleOfTheFiveArmiesExecutable = nil
        }
        
        // 5. Decoder
        let decoderRes = loadDecoderAndGetFinalImage(xIn: latent)
        completion(decoderRes.cgImage, 1.0, "Cooling down...")
        let tock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", tock - tick)
        print("Time", runtime)
        
        if (saveMemory) {
            // reload the unet and text guidance
            initAnUnexpectedJourney()
            initTheDesolationOfSmaug()
            initTheBattleOfTheFiveArmies()
        }
    }
}
