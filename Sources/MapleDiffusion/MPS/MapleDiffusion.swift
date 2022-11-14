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
    let synchronize: Bool
    let modelLocation: URL
    
    private var _textGuidance: TextGuidance?
    var textGuidance: TextGuidance {
        if let _textGuidance {
            return _textGuidance
        }
        let textGuidance = TextGuidance(
            synchronize: synchronize,
            modelLocation: modelLocation,
            device: graphDevice
        )
        _textGuidance = textGuidance
        return textGuidance
    }
    
    private var _uNet: UNet?
    var uNet: UNet {
        if let _uNet {
            return _uNet
        }
        let uNet = UNet(
            synchronize: synchronize,
            modelLocation: modelLocation,
            saveMemory: saveMemory,
            device: graphDevice,
            shape: [1, height, width, 4]
        )
        _uNet = uNet
        return uNet
    }
    
    lazy var diffuser: Diffuser = {
        Diffuser(
            synchronize: synchronize,
            modelLocation: modelLocation,
            device: graphDevice,
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
        synchronize = !device.hasUnifiedMemory
    }
    
    private func randomLatent(seed: Int) -> MPSGraphTensorData {
        let graph = MPSGraph(synchronize: synchronize)
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
        let decoderGraph = MPSGraph(synchronize: synchronize)
        let decoderIn = decoderGraph.placeholder(shape: x.shape, dataType: MPSDataType.float16, name: nil)
        let decoderOut = decoderGraph.makeDecoder(at: modelLocation, xIn: decoderIn)
        return decoderGraph.run(with: commandQueue, feeds: [decoderIn: x], targetTensors: [decoderOut], targetOperations: nil)[decoderOut]!
    }
    
    private func initLatent(image: CGImage, tEnc: Int, timesteps: MPSGraphTensorData, seed: Int) -> MPSGraphTensorData {
        // MEM-HACK: encoder is loaded from disc and deallocated to save memory (at cost of latency)
        
        let imageData = MPSGraphTensorData(device: graphDevice, cgImage: image)
        
        let graph = MPSGraph(synchronize: synchronize)
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
    
    private func generateLatent(
        input: SampleInput,
        completion: @escaping (CGImage?, Float, String)->()
    ) -> MPSGraphTensorData {
        completion(input.initImage, 0, "Tokenizing...")
        
        // 1. String -> Embedding
        let guidanceScaleData = input.guidanceScale.tensorData(device: graphDevice)
        
        let (baseGuidance, textGuidance) = runTextGuidance(prompt: input.prompt, negativePrompt: input.negativePrompt)
        completion(input.initImage, 0.5 * 1 / Float(input.steps), "Generating noise...")
        
        // 2. Noise generation
        let scheduler = Scheduler(synchronize: synchronize, modelLocation: modelLocation, device: graphDevice, steps: input.steps)
        var startStep: Int
        var latent: MPSGraphTensorData
        if let image = input.initImage, let strength = input.strength {
            startStep = Int(Float(input.steps) * strength)
            latent = initLatent(image: image, tEnc: startStep, timesteps: scheduler.timestepsData, seed: input.seed)
        } else {
            startStep = input.steps
            latent = randomLatent(seed: input.seed)
        }
        
        completion(nil, 0.75 * 1 / Float(input.steps), "Starting diffusion...")
        // 3. Diffusion
        for (index, timestep) in scheduler.timesteps[0..<startStep].enumerated().reversed() {
            let tick = CFAbsoluteTimeGetCurrent()
            
            let temb = scheduler.run(with: commandQueue, timestep: timestep)
            let (etaUncond, etaCond) = uNet.run(
                with: commandQueue,
                latent: latent,
                baseGuidance: baseGuidance,
                textGuidance: textGuidance,
                temb: temb
            )
            let (newLatent, auxOut) = diffuser.run(
                with: commandQueue,
                latent: latent,
                timestep: timestep,
                timestepSize: scheduler.timestepSize,
                etaUncond: etaUncond,
                etaCond: etaCond,
                guidanceScale: guidanceScaleData
            )
            latent = newLatent
            
            // update ui
            let tock = CFAbsoluteTimeGetCurrent()
            let stepRuntime = String(format:"%.2fs", tock - tick)
            let progressDesc = index == 0 ? "Decoding..." : "Step \(scheduler.timesteps.count - index) / \(scheduler.timesteps.count) (\(stepRuntime) / step)"
            let outImage = auxOut?.cgImage
            completion(outImage, Float(scheduler.timesteps.count - index) / Float(scheduler.timesteps.count), progressDesc)
        }
        
        if saveMemory {
            // MEM-HACK: unload the unet to fit the decoder
            _uNet = nil
        }
        
        return latent
    }
    
    public func generate(
        input: SampleInput,
        completion: @escaping (CGImage?, Float, String)->()
    ) {
        let tick = CFAbsoluteTimeGetCurrent()
        let latent = generateLatent(input: input, completion: completion)
        
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
    }
}
