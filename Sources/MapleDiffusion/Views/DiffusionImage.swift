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
                progress: Binding<Double>,
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
        VStack {
            if let image {
                
                Image(image, scale: 1, label: Text(label))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
//                    .blur(radius: inputImage == nil ? ((1 - sqrt(progress)) * 600) : 0 )
                    .animation(.default, value: progress)
                    .clipShape(Rectangle())
#if os(macOS)
                    .draggable(enabled: enableDrag, image: image)
#endif
            }
        }
#if os(macOS)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .imageDroppable() { image in
            print("di got image", image, "now setting to image")
            self.inputImage?.wrappedValue = image.scale(targetSize: CGSize(width: 512, height: 512))
            self.image = inputImage?.wrappedValue
        }
#endif
    }
}

