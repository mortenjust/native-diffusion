
import Foundation
//import AppKit
import CoreImage
import CoreGraphics

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
