//
//  MPSGraph+UNet.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

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
