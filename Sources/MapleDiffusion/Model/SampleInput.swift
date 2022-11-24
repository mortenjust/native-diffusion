//
//  SampleInput.swift
//  
//
//  Created by Guillermo Cique Fernández on 14/11/22.
//

import Foundation
import CoreGraphics

public struct SampleInput {
    var prompt: String
    var negativePrompt: String
    var initImage: CGImage? { didSet { checkSize() }}
    var strength: Float?
    var seed: Int
    var steps: Int
    var guidanceScale: Float
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        seed: Int = Int.random(in: 0...Int.max),
        steps: Int = 20,
        guidanceScale: Float = 7.5
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = nil
        self.strength = nil
        self.seed = seed
        self.steps = steps
        self.guidanceScale = guidanceScale
    }
    
    public init(
        prompt: String,
        negativePrompt: String = "",
        initImage: CGImage?,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int.max),
        steps: Int = 20,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage
        self.strength = strength
        self.seed = seed
        self.steps = steps
        self.guidanceScale = guidanceScale
    }
    
    private func checkSize() {
        guard let initImage else { return }
        if initImage.width != 512 || initImage.height != 512 {
            assertionFailure("Please make sure your input image is exactly 512x512. You can use your own cropping mechanism, or the extension in NSImage:crop:to (macOS). Feel free to contribute a general solution. See Github issues for ideas.")
        }
    }
}

#if os(iOS)
import UIKit

public extension SampleInput {
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: UIImage,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int.max),
        steps: Int = 50,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImage = initImage.cgImage!
        self.strength = strength
        self.seed = seed
        self.steps = steps
        self.guidanceScale = guidanceScale
    }
}
#endif

#if os(macOS)
import AppKit

public extension SampleInput {
    init(
        prompt: String,
        negativePrompt: String = "",
        initImage: NSImage,
        strength: Float = 0.75,
        seed: Int = Int.random(in: 0...Int.max),
        steps: Int = 50,
        guidanceScale: Float = 5.0
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        var imageRect = CGRect(x: 0, y: 0, width: initImage.size.width, height: initImage.size.height)
        self.initImage = initImage.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!
        self.strength = strength
        self.seed = seed
        self.steps = steps
        self.guidanceScale = guidanceScale
    }
}
#endif
