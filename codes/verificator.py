import json
from .face_detector import Face_detector
from .mask_detector import Mask_detector
from .headpose_detector import Headpose_detector

class Verificator:
    def __init__(self, img_size, config_path) -> None:
        self.img_size = img_size
        self._initialization(config_path)
    
    def _initialization(self, config_path):
        self.config = self._load_config(config_path)
        self.detectors = {'face': None, 'mask': None, 'headpose': None}
        self.detectors['face'] = Face_detector('./weights/scrfd.onnx')
        # self.detectors['mask'] = Mask_detector()
        # self.detectors['headpose'] = Headpose_detector()
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def verify(self, img):
        '''
        0. 有沒有臉
        1. 臉夠不夠大
        2. 臉有無正對鏡頭
        3. 口罩有無
        ------------------
        result = {
                "face_detected": boolean -  有無偵測到人臉
                "too_many_face" : boolean - 大於一個人臉
                "face_size": str -          臉夠不夠大 (small / good / big)
                "has_mask": boolean -       有無戴口罩
                "headpose": boolean -       臉有無正對鏡頭
                }
        '''
        result = {"face_detected": False, "too_many_face": False, "face_size": None, "has_mask": False, "headpose": False}
        if (img.shape[1], img.shape[0]) != self.img_size:
            raise ValueError(f"Image size must be {self.img_size}")
        
        face_infos = self.detectors['face'].detect(img)
        
        if len(face_infos) == 1:
            face_info = face_infos[0]
            result["face"] = True
            
            face_size_result = self._face_size_verify(img, face_info)
            result["face_size"] = face_size_result
            
            mask_result = self._mask_verify(img, face_info)
            result["mask"] = mask_result
            
            headpose = self._headpose_verify(img, face_info)
            result["headpose"] = headpose
        
            return result, face_info

        else: return result, None
    
    def _face_size_verify(self, img, face):
        return None

    def _headpose_verify(self, img, headpose):
        return None

    def _mask_verify(self, img, mask):
        return None

    