//
//  File.swift
//  
//
//  Created by Morten Just on 10/25/22.
//



#if os(macOS)
import Foundation
import AppKit
import CoreGraphics
import CoreImage
import UniformTypeIdentifiers


extension CGImage {
    func itemProvider(filename: String? = nil) -> NSItemProvider? {
        let rep = NSBitmapImageRep(cgImage: self)
        guard let data = rep.representation(using: .png, properties: [:]) else { return nil }
        
        let filename = filename ?? "Generated Image \(UUID().uuidString.prefix(4))"
        
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent(filename).appendingPathExtension("png")
        do {
            try data.write(to: tempUrl)
            
            let item = NSItemProvider(item: tempUrl as NSSecureCoding, typeIdentifier: UTType.fileURL.identifier)
            item.suggestedName = "\(filename).png"
            
            return item
        } catch {
            return nil
        }
    }
}
#endif
