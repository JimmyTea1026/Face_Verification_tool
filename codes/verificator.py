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
        self.detectors['headpose'] = Headpose_detector('./weights/headpose.onnx')
        # self.detectors['mask'] = Mask_detector()
    
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
            result["face_detected"] = True
            
            face_size_result, face_size_info = self._face_size_verify(img, face_info)
            result["face_size"] = face_size_result
            
            mask_result = self._mask_verify(img, face_info)
            result["mask"] = mask_result
            
            headpose_result, headpose_info = self._headpose_verify(img, face_info)
            result["headpose"] = headpose_result
        
            draw_info = {"face_info": face_info, "face_size_info": face_size_info, "headpose_info": headpose_info}
            return result, draw_info
        
        elif len(face_infos) > 1:
            result["too_many_face"] = True

        return result, None
    
    def _face_size_verify(self, img, face_info):
        all_img_size = img.shape[0] * img.shape[1]
        face_size = face_info['w'] * face_info['h']
        face_size_percent = round(face_size / all_img_size, 2)*100
        min_limit = self.config['face_size_min']
        max_limit = self.config['face_size_max']
        
        result = ""
        if face_size_percent < min_limit: result = 'small'
        elif face_size_percent > max_limit: result = 'big'
        else: result = 'good'
        
        return result, face_size_percent

    def _headpose_verify(self, img, face_info):
        '''
        yaw : 左右擺頭
        pitch : 上下擺頭
        roll : 側扭頭
        '''
        bbox = [face_info['x'], face_info['y'], face_info['w'], face_info['h']]
        result = self.detectors['headpose'].detect(img, bbox)
        
        return True, result

    def _mask_verify(self, img, mask):
        
        return None

    