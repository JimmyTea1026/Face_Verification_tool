import os
import sys
import numpy as np  
import cv2
from .face_detector import Face_detector
from .mask_detector import Mask_detector
from .headpose_detector import Headpose_detector

class Verificator:
    def __init__(self, config) -> None:
        self.config = config
        self.detectors = {'face': None, 'mask': None, 'headpose': None}
        self.detectors['face'] = Face_detector('./weights/scrfd.onnx')
        self.detectors['headpose'] = Headpose_detector('./weights/headpose.onnx')
        self.detectors['mask'] = Mask_detector('./weights/mask.onnx')
        self.img_size = tuple(self.config['img_size'])
    
    def verify(self, img_path:str)->dict:
        '''
        Return : 辨識結果(dict), 繪圖資訊(dict)
        ------------------
        0. 有沒有臉
        1. 臉夠不夠大
        2. 臉有無正對鏡頭
        3. 口罩有無
        ------------------
        result = {
                "error" : str -             Error message, default is None
                "many_face" : boolean -     大於一個人臉
                "face_detected": boolean -  有無偵測到人臉
                "face_size": str -          臉夠不夠大 (small / good / big)
                "with_mask": boolean -      有無戴口罩
                "headpose": boolean -       臉有無正對鏡頭
                "position": boolean -       臉有無處在畫面中央
                }
        '''
        result = {"error": None, "many_face": False, "face_detected": False, "face_size": "", "with_mask": False, "headpose": False, "position": False}
        
        if not os.path.isfile(img_path):
            result['error'] = "File not found"
            return result, None
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
                
        face_infos = self.detectors['face'].detect(img)
        
        if len(face_infos) == 1:    # 只有一個人臉才做後續偵測
            face_info = face_infos[0]
            draw_info = {"face_info": "", "face_size_info": "", 
                "headpose_info": "", "mask_info": "", "position": ""}
            
            draw_info['face_info'] = face_info
            result["face_detected"] = True
            
            face_size_result, face_size_info = self._face_size_verify(img.copy(), face_info)
            result["face_size"] = face_size_result
            draw_info[f"face_size_info"] = face_size_info
            
            headpose_result, headpose_info = self._headpose_verify(img, face_info)
            result["headpose"] = headpose_result
            draw_info["headpose_info"] = headpose_info
            
            position_result, position_info = self._check_face_position(face_info)
            result['position'] = position_result
            draw_info['position_info'] = str(position_info)
            
            if headpose_result: # 因為側臉會讓口罩偵測失準，只有正對鏡頭才做口罩偵測
                mask_result, mask_info = self._mask_verify(img, face_info)
                result["with_mask"] = mask_result
                draw_info["mask_info"] = mask_info
        
            return result, draw_info
        
        else:
            if len(face_infos) > 1:   # 多於一個人臉
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
            return True, mask_info
        else:
            return False, mask_info

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
        iou = round(iou, 2)
        
        if iou > self.config['iou_limit']:
            return True, iou  
        else:
            return False, iou