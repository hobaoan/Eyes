//
//  Measure.swift
//  Eyes
//
//  Created by Hồ Bảo An on 20/12/2023.
//

import UIKit

protocol CameraDelegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int)
}
// Performance Measurement
class Camera {
    
    var delegate: CameraDelegate?
    
    var index: Int = -1
    var measurements: [Dictionary<String, Double>]
    
    init() {
        let measurement = [
            "start": CACurrentMediaTime(),
            "end": CACurrentMediaTime()
        ]
        measurements = Array<Dictionary<String, Double>>(repeating: measurement, count: 30)
    }
    
    // start
    func startCamera() {
        index += 1
        index %= 30
        measurements[index] = [:]
        
        tagLabel(for: index, with: "start")
    }
    
    // stop
    func stopCamera() {
        tagLabel(for: index, with: "end")
        
        let beforeMeasurement = getBeforeMeasurment(for: index)
        let currentMeasurement = measurements[index]
        if let startTime = currentMeasurement["start"],
            let endInferenceTime = currentMeasurement["endInference"],
            let endTime = currentMeasurement["end"],
            let beforeStartTime = beforeMeasurement["start"] {
            delegate?.updateMeasure(inferenceTime: endInferenceTime - startTime,
                                    executionTime: endTime - startTime,
                                    fps: Int(1/(startTime - beforeStartTime)))
        }
        
    }
    
    // labeling with
    func tagLabel(with msg: String? = "") {
        tagLabel(for: index, with: msg)
    }
    
    private func tagLabel(for index: Int, with msg: String? = "") {
        if let message = msg {
            measurements[index][message] = CACurrentMediaTime()
        }
    }
    
    private func getBeforeMeasurment(for index: Int) -> Dictionary<String, Double> {
        return measurements[(index + 30 - 1) % 30]
    }
    
}

class MeasureLogView: UIView {
    let etimeLabel = UILabel(frame: .zero)
    let fpsLabel = UILabel(frame: .zero)
    
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
}
