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
    var inputImage : Binding<CGImage?>?
    @State var isTargeted = false
    
    let label : String
    let draggable : Bool
    
    public init(image: Binding<CGImage?>,
                inputImage: Binding<CGImage?>? = nil,
                progress: Binding<Double>,c
                label: String = "Generated Image",
                draggable: Bool = true)
    {
        self._image = image
        self._progress = progress
        self.inputImage = inputImage
        self.label = label
        self.draggable = draggable
    }
    
    var enableDrag : Bool {
        guard draggable else { return false }
        return progress == 1 ? true : false
    }
    
    public var body: some View {
        
        ZStack {
            if let i = inputImage?.wrappedValue {
                Image(i, scale: 1, label: Text("Input image"))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            if let image {
                Image(image, scale: 1, label: Text(label))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .animation(nil)
                    .blur(radius: (1 - sqrt(progress)) * 100 )
                    .blendMode(progress < 1 ? .sourceAtop : .normal)
                    .animation(.linear(duration: 1), value: progress)
                    .clipShape(Rectangle())
#if os(macOS)
                    .draggable(enabled: enableDrag, image: image)
#endif
            }
        }
           
#if os(macOS)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .imageDroppable() { image in
            let ns = NSImage(cgImage: image, size: .init(width: image.width, height: image.height))
            let cropped = ns.crop(to: .init(width: 512, height: 512))
            let cg = cropped.cgImage
            self.inputImage?.wrappedValue = cg
            self.image = nil
        }
#endif
    }
}

