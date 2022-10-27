//
//  File.swift
//  
//
//  Created by Morten Just on 10/21/22.
//

import Foundation
import SwiftUI
import UniformTypeIdentifiers
/// Displays a CGImage and blurs it according to its generation progress
public struct DiffusionImage : View {
    @Binding var image : CGImage?
    @Binding var progress : Double
    let label : String
    let draggable : Bool
    
    public init(image: Binding<CGImage?>,
                progress: Binding<Double>,
                label: String = "Generated Image",
                draggable: Bool = true)
 {
        self._image = image
        self._progress = progress
        self.label = label
        self.draggable = draggable
    }
    
    var enableDrag : Bool {
        guard draggable else { return false }
        return progress == 1 ? true : false
    }
    
    public var body: some View {
        if let image {
            Image(image, scale: 1, label: Text(label))
                .resizable()
                .aspectRatio(contentMode: .fit)            
                .blur(radius: (1 - sqrt(progress)) * 600 )
                .animation(.default, value: progress)
                .clipShape(Rectangle())
            #if os(macOS)
                .draggable(enabled: enableDrag, image: image)
            #endif
        }
    }
}

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

extension View {
    public func draggable(enabled: Bool, image: CGImage) -> some View {
        self.modifier(FinderDraggable(image: image, enabled: enabled))
    }
}
#endif
