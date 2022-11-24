//
//  Image resizing and cropping for input images
//  
//
//  Created by Morten Just on 11/22/22.
//


#if os(macOS)
import Foundation
import AppKit

extension NSImage {
    var isLandscape : Bool { size.height < size.width }
    func scale(factor: CGFloat) -> NSImage {
        let newWidth = self.size.width * factor
        let newHeight = self.size.height * factor
        let newSize = NSSize(width: newWidth, height: newHeight)
        
        // Draw self into a new image with the new size
        let newImage = NSImage(size: newSize, flipped: false) { rect in
            self.draw(in: .init(x: 0, y: 0, width: newWidth, height: newHeight))
            return true
        }
        return newImage
    }
    
    @objc var cgImage: CGImage? {
           get {
                guard let imageData = self.tiffRepresentation else {
                    return nil }
                guard let sourceData = CGImageSourceCreateWithData(imageData as CFData, nil) else { return nil }
                return CGImageSourceCreateImageAtIndex(sourceData, 0, nil)
           }
        }
    
    func crop(to newSize:NSSize) -> NSImage {
        
        // scale
        var factor : CGFloat = 1
        if isLandscape {
            factor = newSize.height / self.size.height
        } else {
            factor = newSize.width / self.size.width
        }
        let scaledImage = scale(factor: factor)
        
        /// Find the center crop rect
        var fromRect = NSRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        if self.isLandscape {
            fromRect.origin.x = (0.5 * scaledImage.size.width) - (0.5 * newSize.width)
        } else {
            fromRect.origin.y = (0.5 * scaledImage.size.height) - (0.5 * newSize.height)
        }
        
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(newSize.width),
            pixelsHigh: Int(newSize.height),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
            ) else {
                preconditionFailure()
        }
        
        /// Get rid of retina pixels
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        scaledImage.draw(at: .zero, from: fromRect, operation: .sourceOver, fraction: 1.0)
        NSGraphicsContext.restoreGraphicsState()
        
        let data = rep.representation(using: .tiff, properties: [:])
        let newImage = NSImage(data: data!)
        
        return newImage!
    
    }
}



#endif
