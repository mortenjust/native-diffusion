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
    
    public convenience init(device: MPSGraphDevice, cgImage: CGImage) {
        let shape: [NSNumber] = [NSNumber(value: cgImage.height), NSNumber(value: cgImage.width), 3]
        let nsData = cgImage.dataProvider!.data! as NSData
        print("bitsPerPixel", cgImage.bitsPerPixel)
        print("bytesPerRow", cgImage.bytesPerRow)
        print("byteOrderInfo", cgImage.byteOrderInfo.rawValue)
        print("colorSpace", cgImage.colorSpace)
        // Remove alpha channel
        let data = Data(stride(from: 0, to: nsData.count, by: 4).flatMap { i -> [UInt8] in
            [nsData[i], nsData[i+1], nsData[i+2]]
        })
        self.init(device: device, data: data, shape: shape, dataType: .uInt8)
    }
}

/*
 bitsPerPixel 32
 bytesPerRow 2048
 byteOrderInfo CGImageByteOrderInfo
 colorSpace Optional(<CGColorSpace 0x60000006cd80> (kCGColorSpaceICCBased; kCGColorSpaceModelRGB; sRGB IEC61966-2.1))
 */
