//
//  ContentView.swift
//  Single Line Diffusion
//
//  Created by Morten Just on 10/28/22.
//

import SwiftUI
import MapleDiffusion

struct ContentView: View {
    @State var image : CGImage?

    var body: some View {
        VStack {
            if let image { Image(image, scale: 1, label: Text("Generated")) } else { Text("Loading")}
        }
        .onAppear {
            Task.detached {
                for _ in 0...10 {
                    
                    // Local, offline Stable Diffusion in Swift. No Python.
                    // Download + init + generate = 1 line of code:
                    
                    image = try? await Diffusion.generate(localOrRemote: modelUrl, prompt: "cat astronaut")
                }
            }
        }
        .frame(minWidth: 500, minHeight: 500)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}


let modelUrl = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Photato/bins")
