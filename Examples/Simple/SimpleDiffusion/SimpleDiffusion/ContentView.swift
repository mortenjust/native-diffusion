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
    
    @StateObject var sd = Diffusion()
    @State var prompt = ""
    @State var image : CGImage?
    @State var imagePublisher = Diffusion.placeholderPublisher
    @State var progress : Double = 0
    
    var anyProgress : Double { sd.loadingProgress < 1 ? sd.loadingProgress : progress }

    var body: some View {
        VStack {
            DiffusionImage(image: $image, progress: $progress)
            Spacer()
            TextField("Prompt", text: $prompt)
                .onSubmit { self.imagePublisher = sd.generate(prompt: prompt) }
                .disabled(!sd.isModelReady)
            ProgressView(value: anyProgress)
                .opacity(anyProgress == 1 || anyProgress == 0 ? 0 : 1)
        }
        .task { try! await
            sd.prepModels(remoteURL: URL(string: "http://localhost:8080/Diffusion.zip")!) }
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
