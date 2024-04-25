import onnx
import onnxruntime as ort
from .utils.headpose.HeadPoseEstimator import *

class Headpose_detector:
    def __init__(self, modelPath):
        self.estimator = HeadPoseEstimator(modelPath)
        
    def detect(self, img, bbox):
        '''
        Input parameters:
        img : bgr image
        bbox : [x, y, w, h] of face
        
        Return:
        results : {"pitch": pitch, "yaw": yaw, "roll": roll}
        '''
        outputs = self.estimator.run(img, bbox)
        results = {"pitch": outputs[0], "yaw": outputs[1], "roll": outputs[2]}
        return results
