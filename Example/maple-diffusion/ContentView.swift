import SwiftUI
import MapleDiffusion
import Combine

struct ContentView: View {
#if os(iOS)
    let mapleDiffusion : MapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: true)
#else
    @ObservedObject var mapleDiffusion = MapleDiffusion(saveMemoryButBeSlower: false, modelFolder: URL(fileURLWithPath: "/Users/mortenjust/Library/Application Support/Photato/bins"))
    
#endif
    let dispatchQueue = DispatchQueue(label: "Generation")
    @State var steps: Float = 20
    @State var image: Image?
    @State var prompt: String = ""
    @State var negativePrompt: String = ""
    @State var guidanceScale: Float = 7.5
    @State var running: Bool = false
    @State var progressProp: Float = 1
    @State var progressStage: String = "Ready"
    @State var seed : Int = 42
    
    @State var bin = Set<AnyCancellable>()
    
    func generate() {
        dispatchQueue.async {
            running = true
            progressStage = ""
            progressProp = 0
            
            try! mapleDiffusion.generate(prompt: prompt, negativePrompt: negativePrompt, seed: seed, steps: Int(steps), guidanceScale: guidanceScale) { (cgim, p, s) -> () in
                if (cgim != nil) {
                    image = Image(cgim!, scale: 1.0, label: Text("Generated image"))
                }
                progressProp = p
                progressStage = s
            }
            running = false
        }
    }
    
    
    var body: some View {
        VStack {
#if os(iOS)
            Text("🍁 Maple Diffusion").foregroundColor(.orange).bold().frame(alignment: Alignment.center)
#endif
            if (image == nil) {
                Rectangle().fill(.gray).aspectRatio(1.0, contentMode: .fit).frame(idealWidth: mapleDiffusion.width.doubleValue, idealHeight: mapleDiffusion.height.doubleValue)
            } else {
                image!.resizable().aspectRatio(contentMode: .fit).frame(idealWidth: mapleDiffusion.width.doubleValue, idealHeight: mapleDiffusion.height.doubleValue)
            }
            HStack {
                Text("Prompt").bold()
                TextField("What you want", text: $prompt)
            }
            HStack {
                Text("Negative Prompt").bold()
                TextField("What you don't want", text: $negativePrompt)
            }
            HStack {
                HStack {
                    HStack {
                        Text("Scale").bold()
                        Text(String(format: "%.1f", guidanceScale)).foregroundColor(.secondary)
                    }.frame(width: 96, alignment: .leading)
                    Slider(value: $guidanceScale, in: 1...20)
                }
                HStack {
                    Text("Seed").bold()
                    TextField("", value: $seed, format: .number)
                        .frame(maxWidth: 180)
                    Button {
                        seed = Int.random(in: 1...Int.max)
                    } label: {
                        Image(systemName: "arrow.clockwise.circle.fill")
                    }.buttonStyle(.plain)
                }
            }
            HStack {
                HStack {
                    Text("Steps").bold()
                    Text("\(Int(steps))").foregroundColor(.secondary)
                }.frame(width: 96, alignment: .leading)
                Slider(value: $steps, in: 5...150)
            }
            
            ProgressView(progressStage, value: progressProp, total: 1).opacity(running ? 1 : 0).foregroundColor(.secondary)
            
            
            Spacer(minLength: 8)
            Button(action: generate) {
                Text("Generate Image")
                    .frame(minWidth: 100, maxWidth: .infinity, minHeight: 64, alignment: .center)
                    .background(running ? .gray : .blue)
                    .foregroundColor(.white)
                    .font(Font.title)
                    .cornerRadius(32)
            }.buttonStyle(.borderless)
                .disabled(running || !mapleDiffusion.isModelLoaded)
            
        }.padding(16)
            .onReceive(mapleDiffusion.state) { newState in
                switch newState {
                
                case .modelIsLoading(progress: let progress):
                    running = true
                    print("model loading", progress.message, progress.fractionCompleted)
                    progressProp = Float( progress.fractionCompleted)
                    progressStage = progress.message
                case .ready:
                    print("model ready")
                    running = false
                }
            }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
