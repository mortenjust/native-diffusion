//
//  UNet.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class UNet {
    let synchronize: Bool
    let modelLocation: URL
    let saveMemory: Bool
    let device: MPSGraphDevice
    
    // MEM-HACK: split into subgraphs
    private var unetAnUnexpectedJourneyExecutable: MPSGraphExecutable?
    private var anUnexpectedJourneyShapes = [[NSNumber]]()
    private var unetTheDesolationOfSmaugExecutable: MPSGraphExecutable?
    private var theDesolationOfSmaugShapes = [[NSNumber]]()
    private var theDesolationOfSmaugIndices = [MPSGraphTensor: Int]()
    private var unetTheBattleOfTheFiveArmiesExecutable: MPSGraphExecutable?
    private var theBattleOfTheFiveArmiesIndices = [MPSGraphTensor: Int]()
    
    init(synchronize: Bool, modelLocation: URL, saveMemory: Bool, device: MPSGraphDevice, shape: [NSNumber]) {
        self.synchronize = synchronize
        self.modelLocation = modelLocation
        self.saveMemory = saveMemory
        self.device = device
        
        loadAnUnexpectedJourney(shape: shape)
        loadTheDesolationOfSmaug()
        loadTheBattleOfTheFiveArmies()
    }
    
    private func loadAnUnexpectedJourney(shape: [NSNumber]) {
        let graph = MPSGraph(synchronize: synchronize)
        let xIn = graph.placeholder(shape: shape, dataType: MPSDataType.float16, name: nil)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let tembIn = graph.placeholder(shape: [1, 320], dataType: MPSDataType.float16, name: nil)
        let unetOuts = graph.makeUNetAnUnexpectedJourney(
            at: modelLocation,
            xIn: xIn,
            tembIn: tembIn,
            condIn: condIn,
            name: "model.diffusion_model",
            saveMemory: saveMemory
        )
        let unetFeeds = [xIn, condIn, tembIn].reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        unetAnUnexpectedJourneyExecutable = graph.compile(
            with: device,
            feeds: unetFeeds,
            targetTensors: unetOuts,
            targetOperations: nil,
            compilationDescriptor: nil
        )
        anUnexpectedJourneyShapes = unetOuts.map{$0.shape!}
    }
    
    private func loadTheDesolationOfSmaug() {
        let graph = MPSGraph(synchronize: synchronize)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let placeholders = anUnexpectedJourneyShapes.map{graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)} + [condIn]
        theDesolationOfSmaugIndices.removeAll()
        for i in 0..<placeholders.count {
            theDesolationOfSmaugIndices[placeholders[i]] = i
        }
        let feeds = placeholders.reduce(into: [:], {$0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)})
        let unetOuts = graph.makeUNetTheDesolationOfSmaug(at: modelLocation, savedInputsIn: placeholders, name: "model.diffusion_model", saveMemory: saveMemory)
        unetTheDesolationOfSmaugExecutable = graph.compile(with: device, feeds: feeds, targetTensors: unetOuts, targetOperations: nil, compilationDescriptor: nil)
        theDesolationOfSmaugShapes = unetOuts.map{$0.shape!}
    }
    
    private func loadTheBattleOfTheFiveArmies() {
        let graph = MPSGraph(synchronize: synchronize)
        let condIn = graph.placeholder(shape: [saveMemory ? 1 : 2, 77, 768], dataType: MPSDataType.float16, name: nil)
        let unetPlaceholders = theDesolationOfSmaugShapes.map{
            graph.placeholder(shape: $0, dataType: MPSDataType.float16, name: nil)
        } + [condIn]
        theBattleOfTheFiveArmiesIndices.removeAll()
        for i in 0..<unetPlaceholders.count {
            theBattleOfTheFiveArmiesIndices[unetPlaceholders[i]] = i
        }
        let feeds = unetPlaceholders.reduce(into: [:], {
            $0[$1] = MPSGraphShapedType(shape: $1.shape!, dataType: $1.dataType)}
        )
        let unetOut = graph.makeUNetTheBattleOfTheFiveArmies(
            at: modelLocation,
            savedInputsIn: unetPlaceholders,
            name: "model.diffusion_model",
            saveMemory: saveMemory
        )
        unetTheBattleOfTheFiveArmiesExecutable = graph.compile(
            with: device,
            feeds: feeds,
            targetTensors: [unetOut],
            targetOperations: nil,
            compilationDescriptor: nil
        )
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
    
    private func runUNet(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        guidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> MPSGraphTensorData {
        var x = unetAnUnexpectedJourneyExecutable!.run(
            with: queue,
            inputs: reorderAnUnexpectedJourney(x: [latent, guidance, temb]),
            results: nil,
            executionDescriptor: nil
        )
        x = unetTheDesolationOfSmaugExecutable!.run(
            with: queue,
            inputs: reorderTheDesolationOfSmaug(x: x + [guidance]),
            results: nil,
            executionDescriptor: nil
        )
        return unetTheBattleOfTheFiveArmiesExecutable!.run(
            with: queue,
            inputs: reorderTheBattleOfTheFiveArmies(x: x + [guidance]),
            results: nil,
            executionDescriptor: nil
        )[0]
    }
    
    private func runBatchedUNet(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        // concat
        var graph = MPSGraph(synchronize: synchronize)
        let bg = graph.placeholder(shape: baseGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let tg = graph.placeholder(shape: textGuidance.shape, dataType: MPSDataType.float16, name: nil)
        let concatGuidance = graph.concatTensors([bg, tg], dimension: 0, name: nil)
        let concatGuidanceData = graph.run(
            with: queue,
            feeds: [
                bg : baseGuidance,
                tg: textGuidance
            ],
            targetTensors: [concatGuidance], targetOperations
            : nil
        )[concatGuidance]!
        // run
        let concatEtaData = runUNet(with: queue, latent: latent, guidance: concatGuidanceData, temb: temb)
        // split
        graph = MPSGraph(synchronize: synchronize)
        let etas = graph.placeholder(shape: concatEtaData.shape, dataType: concatEtaData.dataType, name: nil)
        let eta0 = graph.sliceTensor(etas, dimension: 0, start: 0, length: 1, name: nil)
        let eta1 = graph.sliceTensor(etas, dimension: 0, start: 1, length: 1, name: nil)
        let etaRes = graph.run(
            with: queue,
            feeds: [etas: concatEtaData],
            targetTensors: [eta0, eta1],
            targetOperations: nil
        )
        return (etaRes[eta0]!, etaRes[eta1]!)
    }
    
    
    func run(
        with queue: MTLCommandQueue,
        latent: MPSGraphTensorData,
        baseGuidance: MPSGraphTensorData,
        textGuidance: MPSGraphTensorData,
        temb: MPSGraphTensorData
    ) -> (MPSGraphTensorData, MPSGraphTensorData) {
        if (saveMemory) {
            // MEM-HACK: un/neg-conditional and text-conditional are run in two separate passes (not batched) to save memory
            let etaUncond = runUNet(with: queue, latent: latent, guidance: baseGuidance, temb: temb)
            let etaCond = runUNet(with: queue, latent: latent, guidance: textGuidance, temb: temb)
            return (etaUncond, etaCond)
        } else {
            return runBatchedUNet(with: queue, latent: latent, baseGuidance: baseGuidance, textGuidance: textGuidance, temb: temb)
        }
    }
}

extension MPSGraph {
    func makeTimeEmbed(at folder: URL, xIn: MPSGraphTensor, name: String) -> MPSGraphTensor {
        var x = xIn
        x = makeLinear(at: folder, xIn: x, name: name + ".0", outChannels: 1280)
        x = swish(x)
        return makeLinear(at: folder, xIn: x, name: name + ".2", outChannels: 1280)
    }

    func makeUNetResBlock(at folder: URL, xIn: MPSGraphTensor, embIn: MPSGraphTensor, name: String, inChannels: NSNumber, outChannels: NSNumber) -> MPSGraphTensor {
        var x = xIn
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".in_layers.0")
        x = makeConv(at: folder, xIn: x, name: name + ".in_layers.2", outChannels: outChannels, khw: 3)
        var emb = embIn
        emb = swish(emb)
        emb = makeLinear(at: folder, xIn: emb, name: name + ".emb_layers.1", outChannels: outChannels)
        emb = expandDims(emb, axes: [1, 2], name: nil)
        x = addition(x, emb, name: nil)
        x = makeGroupNormSwish(at: folder, xIn: x, name: name + ".out_layers.0")
        x = makeConv(at: folder, xIn: x, name: name + ".out_layers.3", outChannels: outChannels, khw: 3)
        
        var skip = xIn
        if (inChannels != outChannels) {
            skip = makeConv(at: folder, xIn: xIn, name: name + ".skip_connection", outChannels: outChannels, khw: 1)
        }
        return addition(x, skip, name: nil)
    }


    func makeOutputBlock(at folder: URL, xIn: MPSGraphTensor, embIn: MPSGraphTensor, condIn: MPSGraphTensor, inChannels: NSNumber, outChannels: NSNumber, dHead: NSNumber, name: String, saveMemory: Bool, spatialTransformer: Bool = true, upsample: Bool = false) -> MPSGraphTensor {
        var x = xIn
        x = makeUNetResBlock(at: folder, xIn: x, embIn: embIn, name: name + ".0", inChannels: inChannels, outChannels: outChannels)
        if (spatialTransformer) {
            x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".1", contextIn: condIn, saveMemory: saveMemory)
        }
        if (upsample) {
            x = upsampleNearest(xIn: x)
            x = makeConv(at: folder, xIn: x, name: name + (spatialTransformer ? ".2" : ".1") + ".conv", outChannels: outChannels, khw: 3)
        }
        return x
    }


    func makeUNetAnUnexpectedJourney(at folder: URL, xIn: MPSGraphTensor, tembIn: MPSGraphTensor, condIn: MPSGraphTensor, name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
        let emb = makeTimeEmbed(at: folder, xIn: tembIn, name: name + ".time_embed")
        
        var savedInputs = [MPSGraphTensor]()
        var x = xIn
        
        if (!saveMemory) {
            // need to explicitly batch to avoid shape errors later iirc
            // TODO: did we actually need this
            x = broadcast(x, shape: [condIn.shape![0], x.shape![1], x.shape![2], x.shape![3]], name: nil)
        }
        
        // input blocks
        x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.0.0", outChannels: 320, khw: 3)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.1.0", inChannels: 320, outChannels: 320)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.1.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.2.0", inChannels: 320, outChannels: 320)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.2.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        // downsample
        x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.3.0.op", outChannels: 320, khw: 3, stride: 2)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.4.0", inChannels: 320, outChannels: 640)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.4.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.5.0", inChannels: 640, outChannels: 640)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.5.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        // downsample
        x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.6.0.op", outChannels: 640, khw: 3, stride: 2)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.7.0", inChannels: 640, outChannels: 1280)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.7.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.8.0", inChannels: 1280, outChannels: 1280)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".input_blocks.8.1", contextIn: condIn, saveMemory: saveMemory)
        savedInputs.append(x)
        
        // downsample
        x = makeConv(at: folder, xIn: x, name: name + ".input_blocks.9.0.op", outChannels: 1280, khw: 3, stride: 2)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.10.0", inChannels: 1280, outChannels: 1280)
        savedInputs.append(x)
        
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".input_blocks.11.0", inChannels: 1280, outChannels: 1280)
        savedInputs.append(x)
        
        // middle blocks
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".middle_block.0", inChannels: 1280, outChannels: 1280)
        x = makeSpatialTransformerBlock(at: folder, xIn: x, name: name + ".middle_block.1", contextIn: condIn, saveMemory: saveMemory)
        x = makeUNetResBlock(at: folder, xIn: x, embIn: emb, name: name + ".middle_block.2", inChannels: 1280, outChannels: 1280)
        
        return savedInputs + [emb] + [x]
    }

    func makeUNetTheDesolationOfSmaug(at folder: URL, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> [MPSGraphTensor] {
        var savedInputs = savedInputsIn
        let condIn = savedInputs.popLast()!
        var x = savedInputs.popLast()!
        let emb = savedInputs.popLast()!
        // output blocks
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.0", saveMemory: saveMemory, spatialTransformer: false, upsample: false)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.1", saveMemory: saveMemory, spatialTransformer: false, upsample: false)
        
        // upsample
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.2", saveMemory: saveMemory, spatialTransformer: false, upsample: true)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.3", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 2560, outChannels: 1280, dHead: 160, name: name + ".output_blocks.4", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        return savedInputs + [emb] + [x]
    }

    func makeUNetTheBattleOfTheFiveArmies(at folder: URL, savedInputsIn: [MPSGraphTensor], name: String, saveMemory: Bool = true) -> MPSGraphTensor {
        var savedInputs = savedInputsIn
        let condIn = savedInputs.popLast()!
        var x = savedInputs.popLast()!
        let emb = savedInputs.popLast()!
        // upsample
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 1920, outChannels: 1280, dHead: 160, name: name + ".output_blocks.5", saveMemory: saveMemory, spatialTransformer: true, upsample: true)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 1920, outChannels: 640, dHead: 80, name: name + ".output_blocks.6", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 1280, outChannels: 640, dHead: 80, name: name + ".output_blocks.7", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        // upsample
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 960, outChannels: 640, dHead: 80, name: name + ".output_blocks.8", saveMemory: saveMemory, spatialTransformer: true, upsample: true)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 960, outChannels: 320, dHead: 40, name: name + ".output_blocks.9", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 640, outChannels: 320, dHead: 40, name: name + ".output_blocks.10", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        x = concatTensors([x, savedInputs.popLast()!], dimension: 3, name: nil)
        x = makeOutputBlock(at: folder, xIn: x, embIn: emb, condIn: condIn, inChannels: 640, outChannels: 320, dHead: 40, name: name + ".output_blocks.11", saveMemory: saveMemory, spatialTransformer: true, upsample: false)
        
        // out
        x = makeGroupNormSwish(at: folder, xIn: x, name: "model.diffusion_model.out.0")
        return makeConv(at: folder, xIn: x, name: "model.diffusion_model.out.2", outChannels: 4, khw: 3)
    }
}
