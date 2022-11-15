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
    
    private var _decoder: Decoder?
    var decoder: Decoder {
        if let _decoder {
            return _decoder
        }
        let decoder = Decoder(
            synchronize: synchronize,
            modelLocation: modelLocation,
            device: graphDevice,
            shape: [1, height, width, 4]
        )
        _decoder = decoder
        return decoder
    }
    
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
    
    private func runTextGuidance(prompt: String, negativePrompt: String) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let guidance = textGuidance.run(with: commandQueue, prompt: prompt, negativePrompt: negativePrompt)
        if saveMemory {
            // MEM-HACK unload the text guidance to fit the unet
            _textGuidance = nil
        }
        return guidance
    }
    
    private func initLatent(input: SampleInput, scheduler: Scheduler) -> MPSGraphTensorData {
        if let image = input.initImage, let strength = input.strength {
            let imageData = MPSGraphTensorData(device: graphDevice, cgImage: image)
            let timestepsData = scheduler.timestepsData
            let startStep = Int(Float(input.steps) * strength)
            let encoder = Encoder(
                synchronize: synchronize,
                modelLocation: modelLocation,
                device: graphDevice,
                inputShape: imageData.shape,
                outputShape: [1, height, width, 4],
                timestepsShape: timestepsData.shape,
                seed: input.seed
            )
            return encoder.run(with: commandQueue, image: imageData, step: startStep, timesteps: timestepsData)
        } else {
            let graph = MPSGraph(synchronize: synchronize)
            let out = graph.randomTensor(
                withShape: [1, height, width, 4],
                descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: .float16)!,
                seed: input.seed,
                name: nil
            )
            return graph.run(with: commandQueue, feeds: [:], targetTensors: [out], targetOperations: nil)[out]!
        }
    }
    
    private func sample(
        latent: inout MPSGraphTensorData,
        input: SampleInput,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        scheduler: Scheduler,
        completion: @escaping (CGImage?, Float, String) -> ()
    ) {
        let guidanceScaleData = input.guidanceScale.tensorData(device: graphDevice)
        let actualTimesteps = scheduler.timesteps(strength: input.strength)
        for (index, timestep) in actualTimesteps.enumerated() {
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
            let progressDesc = index == 0 ? "Decoding..." : "Step \(index) / \(actualTimesteps.count) (\(stepRuntime) / step)"
            let outImage = auxOut?.cgImage
            let progress = 0.1 + (Float(index) / Float(actualTimesteps.count)) * 0.8
            completion(outImage, progress, progressDesc)
        }
        
        if saveMemory {
            // MEM-HACK: unload the unet to fit the decoder
            _uNet = nil
        }
    }
    
    private func runDecoder(latent: MPSGraphTensorData) -> CGImage? {
        let decodedLatent = decoder.run(with: commandQueue, xIn: latent)
        if saveMemory {
            // MEM-HACK unload the decoder
            _decoder = nil
        }
        return decodedLatent.cgImage
    }
    
    public func generate(
        input: SampleInput,
        completion: @escaping (CGImage?, Float, String) -> ()
    ) {
        let mainTick = CFAbsoluteTimeGetCurrent()
        
        // 1. String -> Embedding
        completion(input.initImage, 0, "Tokenizing...")
        let (baseGuidance, textGuidance) = runTextGuidance(prompt: input.prompt, negativePrompt: input.negativePrompt)
        
        // 2. Noise generation
        completion(input.initImage, 0.05, "Generating noise...")
        let scheduler = Scheduler(synchronize: synchronize, modelLocation: modelLocation, device: graphDevice, steps: input.steps)
        var latent = initLatent(input: input, scheduler: scheduler)
        
        // 3. Diffusion
        completion(nil, 0.1, "Starting diffusion...")
        sample(
            latent: &latent,
            input: input,
            baseGuidance: baseGuidance,
            textGuidance: textGuidance,
            scheduler: scheduler,
            completion: completion
        )
        
        // 4. Decoder
        let finalImage = runDecoder(latent: latent)
        completion(finalImage, 1.0, "Cooling down...")
        let mainTock = CFAbsoluteTimeGetCurrent()
        let runtime = String(format:"%.2fs", mainTock - mainTick)
        print("Time", runtime)
    }
}
