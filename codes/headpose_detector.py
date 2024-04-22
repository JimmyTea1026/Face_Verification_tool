import onnx
import onnxruntime as ort

class Headpose_detector:
    def __init__(self, modelPath):
        self.model = onnx.load(modelPath)
        
    def detect(self, frame):
        pass
