import cv2
import numpy as np
import onnx
import onnxruntime as ort
from .utils.detection.scrfd import Scrfd

class Face_detector:
    def __init__(self, modelPath):
        self.model = onnx.load(modelPath)    
        self.scrfd = Scrfd(det_size=(640, 640))

    def detect(self, img):
        img_preprocessed = self.scrfd.preprocess(img)
        session = ort.InferenceSession(self.model.SerializeToString())
        original_inf_results = session.run(None, {'data': img_preprocessed})
        face_infos = self.scrfd.postprocess(original_inf_results, img)
        
        return face_infos
    