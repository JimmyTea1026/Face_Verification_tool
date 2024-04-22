from .face_detector import Face_detector
from .mask_detector import Mask_detector
from .headpose_detector import Headpose_detector

class Verificator:
    def __init__(self, img_size, face_ratio, headpose_angle) -> None:
        self.img_size = img_size
        self.face_ratio = face_ratio
        self.headpose_angle = headpose_angle
        self.detectors = self.init_detectors()
    
    def init_detectors(self):
        detectors = {'face': None, 'mask': None, 'headpose': None}
        detectors['face'] = Face_detector('./weights/scrfd.onnx')
        detectors['mask'] = Mask_detector()
        detectors['headpose'] = Headpose_detector()
        
        return detectors
    
    def verify(self, img):
        face_det, headpose_det, mask_det = self._detect(img)
        face_ver = self._face_verify(face_det)
        headpose_ver = self._headpose_verify(headpose_det)
        mask_ver = self._mask_verify(mask_det)
        return [face_ver, headpose_ver, mask_ver]
    
    def _detect(self, img):
        pass
    
    def _face_verify(face):
        pass

    def _headpose_verify(headpose):
        pass
    
    def _mask_verify(mask):
        pass