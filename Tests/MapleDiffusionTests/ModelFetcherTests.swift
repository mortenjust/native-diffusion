//
//  ModelFetcherTests.swift
//  
//
//  Created by Morten Just on 10/24/22.
//

import XCTest
@testable import MapleDiffusion

/**
 
 Warning:
 These tests are very much work in progress, and far from pure. They require set up and behaviors change from time to time. For example, if you run `testRemoteOnly` with no downloaded folders, it will download. On the second run, it wili not download. I guess the next steps would be to mainipulate the file system to create the same test context every time.
 
 */


final class ModelFetcherTests: XCTestCase {
    
    /// To test, start a web server in the folder you're hosting Diffusion.zip (a zip of all the converted model files)
    let remoteUrl = URL(string: "http://localhost:8080/Diffusion.zip")!
    let localUrl = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("com.example.myapp/bins")
    
    
    override func setUpWithError() throws {
    }

    override func tearDownWithError() throws {
    }
    
    
    
    /// In this case, the developer handles the downloadnig or is embedding in the bundle.
    func testFetchLocalOnly() async throws {
        
        /// Here, we're given only a file URL. We'll return an async stream that only outputs one element
        let fetcher = ModelFetcher(local: localUrl)
        
        var url : URL?
        for await status in fetcher.fetch() {
            print("status: ", status)
            if let u = status.url { url = u }
        }
        XCTAssertNotNil(url)
        print("--> got ", url!)
        
        let fileCount = try! FileManager.default.contentsOfDirectory(at: url!, includingPropertiesForKeys: nil).count
        
        XCTAssert(fileCount > 0)
    }
    
    func testRemoteOnly() async throws {
        
        let fetcher = ModelFetcher(remote: remoteUrl)
        
        var url: URL?
        for await status in fetcher.fetch() {
            print("status: ", status)
            if let u = status.url { url = u }
        }
        
        XCTAssertNotNil(url)
        print("--> got", url!)
        
        let fileCount = try! FileManager.default.contentsOfDirectory(at: url!, includingPropertiesForKeys: nil).count
        
        XCTAssert(fileCount > 0)
    }
    
    func testLocalAndRemoteOnCleanSlate() async throws {
        try FileManager.default.contentsOfDirectory(at: localUrl, includingPropertiesForKeys: nil)
            .forEach { url in
                print("deleting", url)
                try FileManager.default.removeItem(at: url)
            }
        try await testLocalAndRemote()
    }
    
    func testLocalAndRemote() async throws {
        
        let fetcher = ModelFetcher(
            local: localUrl,
            remote: remoteUrl)
        
        var url: URL?
        
        for await status in fetcher.fetch() {
            print("status: ", status)
            if let u = status.url { url = u }
        }
        
        XCTAssertNotNil(url)
        
        print("--> Final URL", url!)
        
        let fileCount = try! FileManager.default.contentsOfDirectory(at: url!, includingPropertiesForKeys: nil).count
        
        XCTAssert(fileCount > 0)
    }
    

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
