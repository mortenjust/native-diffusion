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
        let data = rep.representation(using: .png, properties: [:])
        
        let saveTo = filename ?? "Generated Image \(UUID().uuidString.prefix(4))"
        
        let tempUrl =  FileManager.default.temporaryDirectory.appendingPathComponent(saveTo).appendingPathExtension("png")
        try? data?.write(to:tempUrl)
        
        let item = NSItemProvider(item: tempUrl as NSSecureCoding, typeIdentifier: UTType.url.identifier)

        return item
    }
}
#endif
