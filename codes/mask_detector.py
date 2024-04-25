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
        np_face = np.array(face_resized).astype(np.float32)
        preprocessed_face = np_face / 255.0
        data_face = np.expand_dims(preprocessed_face, axis=0)
        return data_face
    
    def detect(self, face):
        data_face = self._preprocess(face)
        ort_session = ort.InferenceSession(self.model.SerializeToString())
        result = ort_session.run(None, {'input_1': data_face})
        result = np.squeeze(result)
        mask, withoutMask = result[0], result[1]
        label = "Mask" if mask > withoutMask else "No Mask"
        posibility = round(max(mask, withoutMask) * 100, 1)
        
        return label, posibility
