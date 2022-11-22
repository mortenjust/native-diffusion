//
//  File.swift
//  
//
//  Created by Morten Just on 11/21/22.
//

import Foundation
import CoreGraphics
import AppKit




extension CGImage {
    func scale(targetSize: CGSize) -> CGImage? {
        let factor = targetSize.height / CGFloat(self.height)
        guard let scaledImage = scale(factor: factor) else { return nil }
        let xOrigin = Int(scaledImage.width/2) - (Int(targetSize.width)/2)
        let cropRect = CGRect(x: CGFloat(xOrigin), y: 0, width: targetSize.width, height: targetSize.height)
        return scaledImage.cropping(to: cropRect)
    }
    
    func scale(factor: CGFloat) -> CGImage? {
       let newWidth = CGFloat(self.width) * factor
       let newHeight = CGFloat(self.height) * factor
       let bitsPerComponent = self.bitsPerComponent
       let bytesPerRow = self.bytesPerRow
       let colorSpace = self.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        

        print("Attempting to create context for w \(Int(newWidth)) h \(Int(newHeight)), factor \(factor), bpc \(bitsPerComponent), bpr \(bytesPerRow)")
        
       guard let context = CGContext(
          data: nil,
          width: Int(newWidth), height: Int(newHeight),
//          bitsPerComponent: bitsPerComponent,
          bitsPerComponent: 8,
          bytesPerRow: 0,
          space: colorSpace,
          bitmapInfo: bitmapInfo.rawValue)
       else {
           print("Can't create that context")
          return nil
       }

       context.draw(self, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))

       return context.makeImage()
    }
}


public extension NSImage {
    
      func resize(targetSize : NSSize) -> NSImage {
          let image = NSImage(size: targetSize)
          image.lockFocus()
          let ctx = NSGraphicsContext.current
          ctx?.imageInterpolation = .high
          
          let smallestSide = min(size.height, size.width)
          
          self.draw(in: NSRect(x: 0, y: 0,
                               width: targetSize.width,
                               height: targetSize.height),
                    from: NSRect(x: 0, y: 0,
                                 width: smallestSide,
                                 height: smallestSide),
                    operation: .copy, fraction: 1)
          
          image.unlockFocus()
          return image
      }
      
      
      static func unscaledBitmapImageRep(forImage image: NSImage) -> NSBitmapImageRep {
          guard let rep = NSBitmapImageRep(
              bitmapDataPlanes: nil,
              pixelsWide: Int(image.size.width),
              pixelsHigh: Int(image.size.height),
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

          NSGraphicsContext.saveGraphicsState()
          NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
          image.draw(at: .zero, from: .zero, operation: .sourceOver, fraction: 1.0)
          NSGraphicsContext.restoreGraphicsState()

          return rep
      }
    
}
