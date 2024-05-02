import onnx
import onnxruntime as ort
import cv2
import numpy as np

class Mask_detector:
    def __init__(self, modelPath):
        self.model = onnx.load(modelPath)
        
    def _preprocess(self, face):
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        norm_face = self._normalize(face_resized).astype(np.float32)
        data_face = np.expand_dims(norm_face, axis=0)
        return data_face
    
    def _normalize(self, face):
        # Normalize the image with imagenet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        norm_face = (face / 255 - mean) / std
        return norm_face
    
    def detect(self, face):
        data_face = self._preprocess(face)
        ort_session = ort.InferenceSession(self.model.SerializeToString())
        result = ort_session.run(None, {'input_1': data_face})
        result = np.squeeze(result)
        mask, withoutMask = result[0], result[1]
        
        return mask, withoutMask
