//
//  File.swift
//  
//
//  Created by Morten Just on 11/21/22.
//

import Foundation
import CoreGraphics
import SwiftUI

#if os(macOS)

/// Allow safe and optional item dragging
struct FinderDraggable : ViewModifier {
    let image: CGImage
    let enabled: Bool
    
    func body(content: Content) -> some View {
        if enabled, let provider = image.itemProvider() {
            content.onDrag {
                provider
            }
        } else {
           content
        }
    }
}

struct FinderImageDroppable : ViewModifier {
    var isTargeted: Binding<Bool>?
    let forceSize: CGSize?
    var onDrop : ((CGImage) -> Void)
    
    func loadImage(url:URL) {
        let nsImage = NSImage(contentsOf: url)
        let cgImage = nsImage?.cgImage(forProposedRect: nil, context: nil, hints: nil)
        
        // resize here if forcesize is there
        
        
                
        if let cgImage {
            onDrop(cgImage)
        }
    }
    
    func body(content: Content) -> some View {
        content
            .droppable(isTargeted: isTargeted) { url in
                loadImage(url: url)
            }
    }
}

struct FinderDroppable : ViewModifier {
    var isTargeted : Binding<Bool>?
    var onDrop : (URL) -> Void
    
    func body(content: Content) -> some View {
        content
            .onDrop(of: [.fileURL], isTargeted: isTargeted) {  providers in
                if let provider = (providers.first { $0.canLoadObject(ofClass: URL.self ) }) {
                    let _ = provider.loadObject(ofClass: URL.self) { reading, error in
                        if let reading {
                            onDrop(reading)
                        }
                        if let error {
                            print("error", error)
                        }
                    }
                    return true
                }
                return false
            }
    }
}

extension View {
    public func draggable(enabled: Bool,
                          image: CGImage) -> some View {
        self.modifier(FinderDraggable(image: image, enabled: enabled))
    }
    
    public func droppable(isTargeted:Binding<Bool>? = nil,
                          onDrop: @escaping (URL)->Void) -> some View {
        self.modifier(FinderDroppable(isTargeted: isTargeted, onDrop: onDrop))
    }
    
    public func imageDroppable(isTargeted: Binding<Bool>? = nil,
                               forceSize: CGSize? = nil,
                               onDrop:  @escaping (CGImage)->Void) -> some View {
        self.modifier(FinderImageDroppable(isTargeted: isTargeted,
                                           forceSize:forceSize,
                                           onDrop: onDrop))
    }
}
#endif
