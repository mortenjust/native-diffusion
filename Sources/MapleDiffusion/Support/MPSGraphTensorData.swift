//
//  MPSGraphTensorData.swift
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
        let shape: [NSNumber] = [NSNumber(value: cgImage.height), NSNumber(value: cgImage.width), 4]
        let data = cgImage.dataProvider!.data! as Data
        self.init(device: device, data: data, shape: shape, dataType: .uInt8)
    }
}

/*
 bitsPerPixel 32
 bytesPerRow 2048
 byteOrderInfo CGImageByteOrderInfo
 colorSpace Optional(<CGColorSpace 0x60000006cd80> (kCGColorSpaceICCBased; kCGColorSpaceModelRGB; sRGB IEC61966-2.1))
 */


extension Int {
    func tensorData(device: MPSGraphDevice) -> MPSGraphTensorData {
        let data = [Int32(self)].withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: device, data: data, shape: [1], dataType: MPSDataType.int32)
    }
}

extension Float {
    func tensorData(device: MPSGraphDevice) -> MPSGraphTensorData {
        let data = [Float32(self)].withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: device, data: data, shape: [1], dataType: MPSDataType.float32)
    }
}
