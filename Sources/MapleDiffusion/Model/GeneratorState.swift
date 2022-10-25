
//  Created by Morten Just on 10/20/22.
//

import Foundation

public enum GeneratorState: Equatable {
    public static func == (lhs: GeneratorState, rhs: GeneratorState) -> Bool {
        switch(lhs, rhs) {
        case (.ready, .ready):
            return true
//        case (.modelIsLoading(progress: <#T##Double#>, message: <#T##String#>))
        case (.notStarted, .notStarted):
            return true
        default:
            return false
        }
    }
    
    case notStarted
    case modelIsLoading(progress: Double, message: String)
    case ready
}


