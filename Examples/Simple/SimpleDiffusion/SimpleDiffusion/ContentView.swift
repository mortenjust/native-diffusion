//
//  ContentView.swift
//  SimpleDiffusion
//
//  Created by Morten Just on 10/21/22.
//

import SwiftUI
import MapleDiffusion
import Combine

struct ContentView: View {
    
    // 1
    @StateObject var sd = Diffusion()
    @State var prompt = ""
    @State var steps = 20
    @State var guidance : Double = 7.5
    @State var image : CGImage?
    @State var inputImage: CGImage?
    @State var imagePublisher = Diffusion.placeholderPublisher
    @State var progress : Double = 0
    
    var anyProgress : Double { sd.loadingProgress < 1 ? sd.loadingProgress : progress }

    var body: some View {
        VStack {
            
            DiffusionImage(image: $image, inputImage: $inputImage, progress: $progress)
            
            Spacer()
            
            TextField("Prompt", text: $prompt)
            // 3
                .onSubmit {
//                    self.imagePublisher = sd.generate(prompt: prompt)
                    
                    let imageToUse = NSImage(contentsOfFile: "/Users/mortenjust/Desktop/resized.png")
                    // ok, so this works, but my cgimage resizing does not. It's too tricky to debug
                    // the most annoying thing is all the time i spent on writing view modifuckingfiers.
                    // I could just change that to NSImage and be done with it. 
                    
                    
                    let input = SampleInput(prompt: prompt,
                                            initImage: imageToUse!.cgImage(forProposedRect: nil, context: nil, hints: nil)!,
                                            steps: steps)
                    self.imagePublisher = sd.generate(input: input)
                    
                }
                .disabled(!sd.isModelReady)
            
            HStack {
                TextField("Steps", value: $steps, formatter: NumberFormatter())
                TextField("Guidance", value: $guidance, formatter: NumberFormatter())
            }
            
            ProgressView(value: anyProgress)
                .opacity(anyProgress == 1 || anyProgress == 0 ? 0 : 1)
        }
        .task {
            // 2
            let path = URL(string: "http://localhost:8080/Diffusion.zip")!
            do {
                try await sd.prepModels(remoteURL: path)
            } catch {
                assertionFailure("Hi, developer. You most likely don't have a local webserver running that serves the zip file with the transformed model files. ")
            }
        }
        
        // 4
        .onReceive(imagePublisher) { r in
            self.image = r.image
            self.progress = r.progress
        }
        .frame(minWidth: 200, minHeight: 200)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}



extension URL {
    static var modelFolder = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Photato/bins")
}
