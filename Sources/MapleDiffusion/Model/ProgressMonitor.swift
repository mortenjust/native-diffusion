////
////
////  Created by Morten Just on 10/20/22.
////
//
//import Foundation
//
//public struct ProgressMonitor : Equatable {
//    public var completed: Double = 0
//    public var total: Double
//    public var message: String = ""
//    public var fractionCompleted : Double { completed/total }
//
//    mutating func update(completed: Double, message:String) {
//        self.completed = completed
//        self.message = message
//    }
//
//    mutating func with(completed:Double, message:String) -> ProgressMonitor {
//        self.completed = completed
//        self.message = message
//        return self
//    }
//
//    mutating func increasing(withMessage message:String) -> ProgressMonitor {
//        self.completed += 1
//        self.message = message
//        return self
//    }
//
//
//}
//
//
///// UI mockup helper
//
//extension ProgressMonitor {
//    public static func example(_ progress : Double) -> ProgressMonitor {
//        ProgressMonitor(completed: progress, total: 1, message: "Loading files")
//    }
//}
