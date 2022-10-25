//
//  File.swift
//  
//
//  Created by Morten Just on 10/24/22.
//

import Foundation
import Combine

extension URLSession {
    func downloadWithProgress(url: URL, progress:((Double)->Void)? = nil) async throws -> URL {
        var downloadTask: URLSessionDownloadTask?
        var bin = Set<AnyCancellable>()
        
        return try await withCheckedThrowingContinuation({ continuation in
            let request = URLRequest(url: url)
            
            downloadTask = URLSession.shared.downloadTask(with: request) { url, response, error in
                if let url {
                    continuation.resume(returning: url)
                }
                
                if let error {
                    continuation.resume(throwing: error)
                }
            }
            
            /// Send progress to closure if we've got one'
            if let progress, let downloadTask {
                downloadTask.publisher(for: \.progress.fractionCompleted)
                    .sink(receiveValue: { p in
                        progress(p)
                    }).store(in: &bin)
            }
            downloadTask?.resume()
        })
    }
}
