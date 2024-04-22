import onnx
import onnxruntime as ort

class Mask_detector:
    def __init__(self, modelPath):
        self.model = onnx.load(modelPath)
        
    def detect(self, frame):
        pass
