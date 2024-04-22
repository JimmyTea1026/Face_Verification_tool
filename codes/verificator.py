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
        self.detectors['mask'] = Mask_detector()
        self.detectors['headpose'] = Headpose_detector()
    
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
                "face": boolean - 有無偵測到人臉
                "face_size": str - 臉夠不夠大 (small / good / big)
                "mask": boolean - 有無戴口罩
                "headpose": boolean - 臉有無正對鏡頭
                }
        '''
        result = {"face": False, "face_size": "", "mask": False, "headpose": False}
        if img.shape != self.img_size:
            raise ValueError(f"Image size must be {self.img_size}")
        
        face_infos = self._face_detect(img)
        if len(face_infos) > 0:
            face_info = self._largest_face(face_infos)
            result["face"] = True
            
            face_size_result = self._face_size_verify(img, face_info)
            result["face_size"] = face_size_result
            
            mask_result = self._mask_verify(img, face_info)
            result["mask"] = mask_result
            
            headpose = self._headpose_verify(img, face_info)
            result["headpose"] = headpose
        
        return result
    
    def _face_detect(self, img):
        face_infos = []
        return face_infos
    
    def _face_size_verify(img, face):
        pass

    def _headpose_verify(img, headpose):
        pass
    
    def _mask_verify(img, mask):
        pass
    
    def _largest_face(face_infos):
        size_list = []
        for face_info in face_infos:
            size = face_info['w'] * face_info['h']
            size_list.append(size)
        max_index = size_list.index(max(size_list))
        
        return face_infos[max_index]