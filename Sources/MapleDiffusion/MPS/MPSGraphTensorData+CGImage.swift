//
//  MPSGraphTensorData+CGImage.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

extension MPSGraphTensorData {
    var cgImage: CGImage? {
        let shape = self.shape.map{ $0.intValue }
        var imageArrayCPUBytes = [UInt8](repeating: 0, count: shape.reduce(1, *))
        self.mpsndarray().readBytes(&imageArrayCPUBytes, strideBytes: nil)
        return CGImage(
            width: shape[2],
            height: shape[1],
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: shape[2]*shape[3],
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(data: NSData(bytes: &imageArrayCPUBytes, length: imageArrayCPUBytes.count))!,
            decode: nil,
            shouldInterpolate: true,
            intent: CGColorRenderingIntent.defaultIntent
        )
    }
}
