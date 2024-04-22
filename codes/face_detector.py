import onnx
import onnxruntime as ort

class Face_detector:
    def __init__(self, modelPath):
        self.model = onnx.load(modelPath)
        
    def detect(self, img):
        pass
