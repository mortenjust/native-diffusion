
import MetalPerformanceShadersGraph
import Foundation



@available(macOS 12.3, *)
public class MapleDiffusion {
    let device: MTLDevice
    let graphDevice: MPSGraphDevice
    let commandQueue: MTLCommandQueue
    let saveMemory: Bool
    
    // text tokenization
    let tokenizer: BPETokenizer
    
    // text guidance
    var textGuidanceExecutable: MPSGraphExecutable?
    
    // time embedding
    let tembGraph: MPSGraph
    let tembTIn: MPSGraphTensor
    let tembOut: MPSGraphTensor
    
    // diffusion
    let diffGraph: MPSGraph
    let diffGuidanceScaleIn: MPSGraphTensor
    let diffXIn: MPSGraphTensor
    let diffEtaUncondIn: MPSGraphTensor
    let diffEtaCondIn: MPSGraphTensor
    let diffTIn: MPSGraphTensor
    let diffTPrevIn: MPSGraphTensor
    let diffOut: MPSGraphTensor
    let diffAuxOut: MPSGraphTensor
    
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
    
    
    public init(saveMemoryButBeSlower: Bool = true, modelFolder mf : URL) {
        // set global folder
        modelFolder = mf
        
        saveMemory = saveMemoryButBeSlower
        device = MTLCreateSystemDefaultDevice()!
        graphDevice = MPSGraphDevice(mtlDevice: device)
        commandQueue = device.makeCommandQueue()!
        
        // text tokenization
        tokenizer = BPETokenizer()
        
        // time embedding
        tembGraph = makeGraph()
        tembTIn = tembGraph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        tembOut = makeTimeFeatures(graph: tembGraph, tIn: tembTIn)
        
        // diffusion
        diffGraph = makeGraph()
        diffXIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        diffEtaUncondIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        diffEtaCondIn = diffGraph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        diffTIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        diffTPrevIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.int32, name: nil)
        diffGuidanceScaleIn = diffGraph.placeholder(shape: [1], dataType: MPSDataType.float16, name: nil)
        diffOut = makeDiffusionStep(graph: diffGraph, xIn: diffXIn, etaUncondIn: diffEtaUncondIn, etaCondIn: diffEtaCondIn, tIn: diffTIn, tPrevIn: diffTPrevIn, guidanceScaleIn: diffGuidanceScaleIn)
        diffAuxOut = makeAuxUpsampler(graph: diffGraph, xIn: diffOut)
        
        // text guidance
        initTextGuidance()
        
        // unet
        initAnUnexpectedJourney()
        initTheDesolationOfSmaug()
        initTheBattleOfTheFiveArmies()
    }
    
    private func initTextGuidance() {
        let graph = makeGraph()
        let textGuidanceIn = graph.placeholder(shape: [2, 77], dataType: MPSDataType.int32, name: nil)
        let textGuidanceOut = makeTextGuidance(graph: graph, xIn: textGuidanceIn, name: "cond_stage_model.transformer.text_model")
        let textGuidanceOut0 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 0, length: 1, name: nil)
        let textGuidanceOut1 = graph.sliceTensor(textGuidanceOut, dimension: 0, start: 1, length: 1, name: nil)
        textGuidanceExecutable = graph.compile(with: graphDevice, feeds: [textGuidanceIn: MPSGraphShapedType(shape: textGuidanceIn.shape, dataType: MPSDataType.int32)], targetTensors: [textGuidanceOut0, textGuidanceOut1], targetOperations: nil, compilationDescriptor: nil)
    }
    
    private func initAnUnexpectedJourney() {
        let graph = makeGraph()
        let xIn = graph.placeholder(shape: [1, height, width, 4], dataType: MPSDataType.float16, name: nil)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let tembIn = graph.placeholder(shape: [1, 320], dataType: MPSDataType.float16, name: nil)
        let unetOuts = makeUNetAnUnexpectedJourney(graph: graph, xIn: xIn, tembIn: tembIn, condIn: condIn, name: "model.diffusion_model", saveMemory: saveMemory)
        let unetFeeds = [xIn, condIn, tembIn].reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        unetAnUnexpectedJourneyExecutable = graph.compile(with: graphDevice, feeds: unetFeeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        anUnexpectedJourneyShapes = unetOuts.map{$0.shape!}
    }
    
    private func initTheDesolationOfSmaug() {
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
    
    private func initTheBattleOfTheFiveArmies() {
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
    
    private func runTextGuidance(baseTokens: [Int], tokens: [Int]) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let tokensData = (baseTokens + tokens).map({Int32($0)}).withUnsafeBufferPointer {Data(buffer: $0)}
        let tokensMPSData = MPSGraphTensorData(device: graphDevice, data: tokensData, shape: [2, 77], dataType: MPSDataType.int32)
        let res = textGuidanceExecutable!.run(with: commandQueue, inputs: [tokensMPSData], results: nil, executionDescriptor: nil)
        return (res[0], res[1])
    }
    
    private func loadDecoderAndGetFinalImage(xIn: MPSGraphTensorData) -> MPSGraphTensorData {
        // MEM-HACK: decoder is loaded from disc and deallocated to save memory (at cost of latency)
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
    
    private func runUNet(latent: MPSGraphTensorData, guidance: MPSGraphTensorData, temb: MPSGraphTensorData) -> MPSGraphTensorData {
        var x = unetAnUnexpectedJourneyExecutable!.run(with: commandQueue, inputs: reorderAnUnexpectedJourney(x: [latent, guidance, temb]), results: nil, executionDescriptor: nil)
        x = unetTheDesolationOfSmaugExecutable!.run(with: commandQueue, inputs: reorderTheDesolationOfSmaug(x: x + [guidance]), results: nil, executionDescriptor: nil)
        return unetTheBattleOfTheFiveArmiesExecutable!.run(with: commandQueue, inputs: reorderTheBattleOfTheFiveArmies(x: x + [guidance]), results: nil, executionDescriptor: nil)[0]
    }
    
    private func runBatchedUNet(latent: MPSGraphTensorData, baseGuidance: MPSGraphTensorData, textGuidance: MPSGraphTensorData, temb: MPSGraphTensorData) -> (MPSGraphTensorData, MPSGraphTensorData) {
        // concat
        var graph = makeGraph()
        let bg = graph.placeholder(shape: baseGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let tg = graph.placeholder(shape: textGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let concatGuidance = graph.concatTensors([bg, tg], dimension: 0, name: nil)
        let concatGuidanceData = graph.run(feeds: [bg : baseGuidance, tg: textGuidance], targetTensors: [concatGuidance], targetOperations: nil)[concatGuidance]!
        // run
        let concatEtaData = runUNet(latent: latent, guidance: concatGuidanceData, temb: temb)
        // split
        graph = makeGraph()
        let etas = graph.placeholder(shape: concatEtaData.shape, dataType: concatEtaData.dataType, name: nil)
        let eta0 = graph.sliceTensor(etas, dimension: 0, start: 0, length: 1, name: nil)
        let eta1 = graph.sliceTensor(etas, dimension: 0, start: 1, length: 1, name: nil)
        let etaRes = graph.run(feeds: [etas: concatEtaData], targetTensors: [eta0, eta1], targetOperations: nil)
        return (etaRes[eta0]!, etaRes[eta1]!)
    }
    
    private func generateLatent(prompt: String, negativePrompt: String, seed: Int, steps: Int, guidanceScale: Float, completion: @escaping (CGImage?, Float, String)->()) -> MPSGraphTensorData {
        completion(nil, 0, "Tokenizing...")
        
        // 1. String -> Tokens
        let baseTokens = tokenizer.encode(s: negativePrompt)
        let tokens = tokenizer.encode(s: prompt)
        completion(nil, 0.25 * 1 / Float(steps), "Encoding...")
        
        // 2. Tokens -> Embedding
        let (baseGuidance, textGuidance) = runTextGuidance(baseTokens: baseTokens, tokens: tokens)
        if (saveMemory) {
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
            let tMPSData = MPSGraphTensorData(device: graphDevice, data: tData, shape: [1], dataType: MPSDataType.int32)
            let tPrevData = [Int32(tsPrev)].withUnsafeBufferPointer {Data(buffer: $0)}
            let tPrevMPSData = MPSGraphTensorData(device: graphDevice, data: tPrevData, shape: [1], dataType: MPSDataType.int32)
            let guidanceScaleData = [Float16(guidanceScale)].withUnsafeBufferPointer {Data(buffer: $0)}
            let guidanceScaleMPSData = MPSGraphTensorData(device: graphDevice, data: guidanceScaleData, shape: [1], dataType: MPSDataType.float16)
            let temb = tembGraph.run(with: commandQueue, feeds: [tembTIn: tMPSData], targetTensors: [tembOut], targetOperations: nil)[tembOut]!
            let etaUncond: MPSGraphTensorData
            let etaCond: MPSGraphTensorData
            if (saveMemory) {
                // MEM-HACK: un/neg-conditional and text-conditional are run in two separate passes (not batched) to save memory
                etaUncond = runUNet(latent: latent, guidance: baseGuidance, temb: temb)
                etaCond = runUNet(latent: latent, guidance: textGuidance, temb: temb)
            } else {
                (etaUncond, etaCond) = runBatchedUNet(latent: latent, baseGuidance: baseGuidance, textGuidance: textGuidance, temb: temb)
            }
            let res = diffGraph.run(with: commandQueue, feeds: [diffXIn: latent, diffEtaUncondIn: etaUncond, diffEtaCondIn: etaCond, diffTIn: tMPSData, diffTPrevIn: tPrevMPSData, diffGuidanceScaleIn: guidanceScaleMPSData], targetTensors: [diffOut, diffAuxOut], targetOperations: nil)
            latent = res[diffOut]!
            
            // update ui
            let tock = CFAbsoluteTimeGetCurrent()
            let stepRuntime = String(format:"%.2fs", tock - tick)
            let progressDesc = t == 0 ? "Decoding..." : "Step \(timesteps.count - t) / \(timesteps.count) (\(stepRuntime) / step)"
            completion(tensorToCGImage(data: res[diffAuxOut]!), Float(timesteps.count - t) / Float(timesteps.count), progressDesc)
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
        completion(tensorToCGImage(data: decoderRes), 1.0, "Cooling down...")
        
        if (saveMemory) {
            // reload the unet and text guidance
            initAnUnexpectedJourney()
            initTheDesolationOfSmaug()
            initTheBattleOfTheFiveArmies()
            initTextGuidance()
        }
    }
}
