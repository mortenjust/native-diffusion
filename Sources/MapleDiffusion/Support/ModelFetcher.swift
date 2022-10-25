//
//  File.swift
//  
//
//  Created by Morten Just on 10/22/22.
//

import Foundation
import Combine
import ZIPFoundation

class ModelFetcher {
    let local: URL?
    let remote: URL?
    
    private var bin = Set<AnyCancellable>()
    
    // Initializer overloads
    
    /// We saved our model locally, e.g. in the bundle.
    init(local: URL) {
        self.local = local
        self.remote = nil
    }
    
    /// (Recommended)   Use the models at the default location. Download it first if not there.
    init(remote: URL) {
        self.remote = remote
        self.local = nil
    }
    
    /// Use the models at the given local location. Download to this destination if not there.
    init(local: URL, remote: URL) {
        self.local = local
        self.remote = remote
    }
    
    struct FileDownload {
        let url : URL?
        let progress: Double
        let stage:String
    }
    
    // Helpers
    
    
    private var bundleId : String {
        Bundle.main.bundleIdentifier ?? "app.otato.diffusion"
    }
    
    var finalLocalUrl: URL {
        local ?? fallbackModelFolder
    }
    
    /// Used if no local URL is provided
    var fallbackModelFolder : URL {
        // TODO: If on iOS use the bundle
        
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(bundleId)
            .appendingPathComponent("bins")
    }
    
    /// Checks final destination and returns true if it's not empty
    var isModelPresentLocally : Bool {
        do {
            let files = try FileManager.default
                .contentsOfDirectory(at: finalLocalUrl, includingPropertiesForKeys: nil)
            return !files.isEmpty
        } catch {
            return false
        }
    }
    
    // methods
    
    enum ModelFetcherError : Error {
        case emptyFolderAndNoRemoteURL
    }

    func fetch(progress: ProgressClosure?) async throws -> URL
    {
        
        let combinedSteps = 2.0 // Download and unzip
        var combinedProgress = 0.0
        
        // is the calculated local folder, and it's not empty? go!
        if isModelPresentLocally {
            progress?(1.0)
            return finalLocalUrl }

        // create the folder if it doesn't exist
        try FileManager.default.createDirectory(at: finalLocalUrl, withIntermediateDirectories: true)
        
        // if the local folder is empty and we have a remote, start downloading and return a stream
        if !isModelPresentLocally, let remote  {
            
            // 1 download
            let downloadedURL = try await URLSession
                .shared
                .downloadWithProgress(url: remote) { p in
                    combinedProgress += (p/combinedSteps)
                    progress?(combinedProgress)
                }
            
            // 2. unzip
            try await moveAndUnzip(downloadedURL) { p in
                combinedProgress += (p/combinedSteps)
                progress?(combinedSteps)
            }
            
            // 3 done!
            return finalLocalUrl
        }
        
        // with the overloads we shouldn't end up here
        assertionFailure()
        throw ModelFetcherError.emptyFolderAndNoRemoteURL
//        fatalError("The model folder is empty or not there, and no remote URL was provided")
    }
 
    
    func moveAndUnzip(_ url : URL, completion: ProgressClosure?) async throws {
        let progress = Progress()
        var bin = Set<AnyCancellable>()
        progress.publisher(for: \.fractionCompleted)
            .map { Double(integerLiteral: $0) }
            .sink { value in
                completion?(value)
        }.store(in: &bin)
        try FileManager.default.unzipItem(at: url, to: finalLocalUrl, progress: progress)
    }
    
    

}


public typealias ProgressClosure = (Double) -> Void
