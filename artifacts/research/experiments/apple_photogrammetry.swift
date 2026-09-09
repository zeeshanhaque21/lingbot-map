import Foundation
import RealityKit

// Native, non-generative comparison against learned-depth reconstruction.
// Compile: swiftc -parse-as-library apple_photogrammetry.swift -o apple-photogrammetry
@main
struct ApplePhotogrammetry {
    static func main() async {
        do {
            try await reconstruct()
        } catch {
            print("error: \(error)\nhelp: Check the image directory and supported output format")
            Foundation.exit(1)
        }
    }

    static func reconstruct() async throws {
        if CommandLine.arguments.dropFirst().contains("--help") {
            print("usage: apple-photogrammetry <image-directory> <new-output.usdz>")
            print("description: Native Mac photogrammetry comparison with object masking disabled")
            return
        }
        guard PhotogrammetrySession.isSupported else {
            print("error: Object Capture is unsupported on this Mac")
            Foundation.exit(1)
        }
        let arguments = CommandLine.arguments
        guard arguments.count == 3 else {
            print("usage: apple-photogrammetry <image-directory> <output.usdz>")
            Foundation.exit(1)
        }
        let input = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let output = URL(fileURLWithPath: arguments[2])
        guard !FileManager.default.fileExists(atPath: output.path) else {
            print("error: Output already exists; choose a new path")
            Foundation.exit(1)
        }
        var configuration = PhotogrammetrySession.Configuration()
        configuration.sampleOrdering = .sequential
        configuration.featureSensitivity = .high
        configuration.isObjectMaskingEnabled = false
        let session = try PhotogrammetrySession(input: input, configuration: configuration)
        try session.process(requests: [.modelFile(url: output, detail: .medium)])
        var lastProgress = -1
        for try await event in session.outputs {
            switch event {
            case .processingComplete:
                print("status: complete\noutput: \(output.path)")
                return
            case .requestError(_, let error):
                print("error: \(error)")
                Foundation.exit(1)
            case .requestProgress(_, let fraction):
                let progress = Int(fraction * 100)
                if progress != lastProgress {
                    print("progress_percent: \(progress)")
                    lastProgress = progress
                }
            case .invalidSample(let id, let reason):
                print("invalid_sample: \(id) \(reason)")
            case .skippedSample(let id):
                print("skipped_sample: \(id)")
            case .inputComplete:
                print("input: complete")
            default:
                break
            }
            fflush(stdout)
        }
        print("error: Session ended without a completed model")
        Foundation.exit(1)
    }
}
