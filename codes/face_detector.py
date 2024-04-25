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
        '''
        Input parameters:
        img : bgr image
        
        Return:
        face_infos : list of dict
        {x, y, w, h, left_eyes_x, left_eyes_y, right_eyes_x, right_eyes_y, 
        nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y, confidence}
        '''
        img_preprocessed = self.scrfd.preprocess(img)
        session = ort.InferenceSession(self.model.SerializeToString())
        original_inf_results = session.run(None, {'data': img_preprocessed})
        face_infos = self.scrfd.postprocess(original_inf_results, img)
        if face_infos is None: face_infos = []
        return face_infos
    