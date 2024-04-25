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
        self.detectors['mask'] = Mask_detector('./weights/mask.onnx')
    
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
                "many_face" : boolean - 大於一個人臉
                "face_size": str -          臉夠不夠大 (small / good / big)
                "no_mask": boolean -       有無戴口罩
                "headpose": boolean -       臉有無正對鏡頭
                }
        '''
        result = {"face_detected": False, "many_face": False, "face_size": None, "no_mask": False, "headpose": False}
        if (img.shape[1], img.shape[0]) != self.img_size:   # 輸入圖片大小不符合，raise error
            raise ValueError(f"Image size must be {self.img_size}")
        
        face_infos = self.detectors['face'].detect(img)
        
        if len(face_infos) == 1:    # 只有一個人臉才做後續偵測
            face_info = face_infos[0]
            result["face_detected"] = True
            
            face_size_result, face_size_info = self._face_size_verify(img.copy(), face_info)
            result["face_size"] = face_size_result
            
            headpose_result, headpose_info = self._headpose_verify(img, face_info)
            result["headpose"] = headpose_result
            
            mask_result, mask_info = self._mask_verify(img, face_info)
            result["no_mask"] = mask_result
            
            position_result = self._check_face_position(face_info)
        
            draw_info = {"face_info": face_info, "face_size_info": face_size_info, "headpose_info": headpose_info, "mask_info": mask_info}
            return result, draw_info
        
        elif len(face_infos) > 1:   # 多於一個人臉
            result["many_face"] = True

        return result, None
    
    def _face_size_verify(self, img, face_info):
        '''
        "人臉bbox所佔的像素" 需達到 "整體圖片像素" 的特定比例，才會輸出 "good" 的結果
        '''
        all_img_size = img.shape[0] * img.shape[1]
        face_size = face_info['w'] * face_info['h']
        face_size_percent = round(face_size / all_img_size, 2)
        min_limit = self.config['face_size_min']
        max_limit = self.config['face_size_max']
        
        result = ""
        if face_size_percent < min_limit: result = 'small'
        elif face_size_percent > max_limit: result = 'big'
        else: result = 'good'
        
        return result, face_size_percent*100

    def _headpose_verify(self, img, face_info):
        '''
        yaw : 左右擺頭
        pitch : 上下擺頭
        roll : 側扭頭
        檢查三者的絕對值是否小於閥值
        '''
        bbox = [face_info['x'], face_info['y'], face_info['w'], face_info['h']]
        result = self.detectors['headpose'].detect(img, bbox)
        pitch, yaw, roll = result['pitch'], result['yaw'], result['roll']
        if abs(pitch) < self.config['pitch_limit'] and abs(yaw) < self.config['yaw_limit'] and abs(roll) < self.config['roll_limit']:
            return True, result
        
        return False, result

    def _mask_verify(self, img, face_info):
        x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
        face = img[y:y+h, x:x+w]
        mask_result, posibility = self.detectors['mask'].detect(face)
        mask_info = f"{mask_result} : {posibility}%"
        if mask_result == 'Mask':
            return False, mask_info
        else:
            return True, mask_info

    def _check_face_position(self, face_info):
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            
            box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
                        
            iou = intersection_area / min(box1_area, box2_area)
            return iou

        box1 = [face_info['x'], face_info['y'], face_info['x']+face_info['w'], face_info['y']+face_info['h']]
        box2 = [self.config['valid_area_x'], self.config['valid_area_y'], 
                self.config['valid_area_x']+self.config['valid_area_w'], self.config['valid_area_y']+self.config['valid_area_h']]
        iou = calculate_iou(box1, box2)
        print("IOU between box1 and box2:", iou)
        return 