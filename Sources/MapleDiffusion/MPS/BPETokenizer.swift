//
//  BPETokenizer.swift
//  
//
//  Created by Guillermo Cique Fernández on 9/11/22.
//

import Foundation
import MetalPerformanceShadersGraph

class BPETokenizer {
    // why didn't they just byte-encode
    func whitespaceClean(s: String) -> String {
        return s.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    func getPairs(s: [String]) -> Set<String> {
        return Set<String>((1..<s.count).map({(s[$0 - 1] + " " + s[$0])}))
    }
    
    let pat: NSRegularExpression = try! NSRegularExpression(pattern: #"'s|'t|'re|'ve|'m|'ll|'d|[^\s]+"#, options: NSRegularExpression.Options.caseInsensitive)
    var bytesToUnicode = [Int:Character]()
    var ranks = [String:Int]()
    var vocab: [String:Int]
    
    public init(modelLocation: URL) {
        var vocabList = [String]()
        for i in Array(33...126) + Array(161...172) + Array(174...255) {
            bytesToUnicode[i] = Character(Unicode.Scalar(i)!)
            vocabList.append(String(Unicode.Scalar(i)!))
        }
        for i in 0...255 {
            if (bytesToUnicode[i] != nil) { continue }
            bytesToUnicode[i] = Character(Unicode.Scalar(256 + bytesToUnicode.count - 188)!)
            vocabList.append(String(bytesToUnicode[i]!))
        }
        vocabList += vocabList.map({$0 + "</w>"})
//        `var vocabFileURL = modelFolder.appendingPathComponent("bpe_simple_vocab_16e6").appendingPathExtension("txt")`
//        let vocabFile = try! String(contentsOf: Bundle.main.url(forResource: "bins/bpe_simple_vocab_16e6", withExtension: "txt")!)
        
        let vocabFile = try! String(
            contentsOf: modelLocation
                .appendingPathComponent("bpe_simple_vocab_16e6")
                .appendingPathExtension("txt")
        )
        
        for (i, m) in vocabFile.split(separator: "\n")[1..<48_895].enumerated() {
            ranks[String(m)] = i
            vocabList.append(m.split(separator: " ").joined(separator: ""))
        }
        vocab = vocabList.enumerated().reduce(into: [:], {$0[$1.element] = $1.offset})
    }
    
    func encodeToken(s: String) -> [Int] {
        let token = String(s.utf8.map{bytesToUnicode[Int($0)]!})
        var word = token[..<token.index(before: token.endIndex)].map{String($0)} + [token.suffix(from: token.index(before: token.endIndex)) + "</w>"]
        var pairs = getPairs(s: Array(word))
        var mergedWordTokens = [token + "</w>"]
        var count = 0
        if (!pairs.isEmpty) {
            while (true) {
                count += 1
                assert(count < 8192, "encodeToken is trapped in a token factory for input \(s)")
                let highestRankedBigram = pairs.min(by: {ranks[$0, default: Int.max] < ranks[$1, default: Int.max]})!
                if (ranks[highestRankedBigram] == nil) { break }
                let fs = highestRankedBigram.split(separator: " ")
                let (first, second) = (String(fs[0]), String(fs[1]))
                var (newWord, i) = ([String](), 0)
                while (i < word.count) {
                    let j = word[i..<word.count].firstIndex(of: first)
                    if (j == nil) {
                        newWord.append(contentsOf: word[i..<word.count])
                        break
                    } else {
                        newWord.append(contentsOf: word[i..<j!])
                        i = j!
                    }
                    if (word[i] == first && word[i + 1] == second) {
                        newWord.append(first + second)
                        i += 2
                    } else {
                        newWord.append(word[i])
                        i += 1
                    }
                }
                word = newWord
                if (word.count == 1) {
                    break
                } else {
                    pairs = getPairs(s: word)
                }
            }
            mergedWordTokens = word
        }
        return mergedWordTokens.map{ vocab[$0]! }
    }
    
    public func encode(s: String) -> [Int] {
        let ns = NSString(string: whitespaceClean(s: s.lowercased()))
        var bpe: [Int] = []
        for match in pat.matches(in: String(ns), range: NSRange(location: 0, length: ns.length)) {
            bpe.append(contentsOf: encodeToken(s: ns.substring(with: match.range)))
        }
        if (bpe.count > 75) {
            print("Prompt of \(bpe.count) bpe tokens will be truncated: \(s)")
        }
        return [49406] + bpe[..<min(75, bpe.count)] + [Int](repeating: 49407, count: max(1, 76 - bpe.count))
    }
}
