import os
import cv2
import math
import numpy as np
from .face_detector import Face_detector
from .mask_detector import Mask_detector
from .headpose_detector import Headpose_detector

def draw_frame(frame, results, draw_info):
    if draw_info is not None:
        face_info = draw_info['face_info']
        headpose_info = draw_info['headpose_info']
        face_size_info = draw_info['face_size_info']
        position_info = draw_info['position_info']
        mask_info = draw_info['mask_info']
        #------------------------------------
        cv2.rectangle(frame, (face_info['x'], face_info['y']), (face_info['x'] + face_info['w'], face_info['y'] + face_info['h']), (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_eye'], 2, (255, 0, 0), 2)
        cv2.circle(frame, face_info['right_eye'], 2, (0, 0, 255), 2)
        cv2.circle(frame, face_info['nose'], 2, (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_mouth'], 2, (255, 0, 255), 2)
        cv2.circle(frame, face_info['right_mouth'], 2, (0, 255, 255), 2)
        cv2.putText(frame, f"conf : {face_info['confidence']:.3f}", (face_info['x']-10, face_info['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        #------------------------------------
        cv2.putText(frame, f"size : {int(face_size_info)}%", (face_info['x']-10, face_info['y'] + face_info['h'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        v_x, v_y, v_w, v_h = 710, 290, 500, 500
        cv2.rectangle(frame, (v_x, v_y), (v_x + v_w, v_y + v_h), (255, 255, 255), 2)
        #------------------------------------
        cv2.putText(frame, f"IOU : {position_info}%", (face_info['x']-10, face_info['y'] + face_info['h'] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        #------------------------------------
        cv2.putText(frame, mask_info, (face_info['x']-10, face_info['y'] + face_info['h'] + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        #------------------------------------
        pitch, yaw, roll = headpose_info['pitch'], headpose_info['yaw'], headpose_info['roll']
        cv2.putText(frame, f"pitch : {(pitch):.1f}", (face_info['x']-10, face_info['y'] + face_info['h'] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(frame, f"yaw : {(yaw):.1f}", (face_info['x']-10, face_info['y'] + face_info['h'] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, f"roll : {(roll):.1f}", (face_info['x']-10, face_info['y'] + face_info['h'] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        tdx = face_info['nose'][0]
        tdy = face_info['nose'][1]
        size = 100
        # X-Axis pointing to right. drawn in red
        x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
        y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
        y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (math.sin(yaw)) + tdx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

        cv2.line(frame, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(frame, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(frame, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)
        
    return frame

class Verificator:
    def __init__(self, config) -> None:
        self.config = config
        self.detectors = {'face': None, 'mask': None, 'headpose': None}
        self.detectors['face'] = Face_detector('./weights/scrfd.onnx')
        self.detectors['headpose'] = Headpose_detector('./weights/headpose.onnx')
        self.detectors['mask'] = Mask_detector('./weights/mask.onnx')
        self.img_size = tuple(self.config['img_size'])
    
    def _preprocess(self, img):
        if type(img) == str:
            img_path = img
            img = cv2.imread(img_path)
        
        img = cv2.resize(img, self.img_size)
        return img
    
    def verify(self, img, with_mask)->dict:
        '''
        result = {
                "put_off_mask": boolean -  True代表需要拿下口罩
                "put_on_mask": boolean -   True代表需要戴上口罩
                "many_face": boolean -     True代表偵測到多於一個人臉
                "small_face": boolean -    臉太小，True代表太小
                "big_face": boolean -      臉太大，True代表太大
                "headpose": boolean -      臉有無正對鏡頭，True代表沒有正對鏡頭
                "no_face": boolean -       沒有偵測到人臉，True代表沒有偵測到人臉
                "position": boolean -      臉有無處在畫面中央，True代表沒有處在畫面中央
                }
        '''
        result = {'put_off_mask': False, 'put_on_mask': False, 'many_face': False, 'small_face': False, 'big_face': False, 'headpose': False, 'no_face': False, 'position': False}
        
        img = self._preprocess(img)
        
        face_infos = self.detectors['face'].detect(img)
        
        if len(face_infos) > 1:   # 多於一個人臉
            result["many_face"] = True
            
        elif len(face_infos) == 0:  # 沒有人臉
            result["no_face"] = True
            
        elif len(face_infos) == 1:    # 只有一個人臉才做後續偵測
            face_info = face_infos[0]
            draw_info = {"face_info": "", "face_size_info": "", 
                "headpose_info": "", "mask_info": "", "position": ""}
            
            draw_info['face_info'] = face_info
            
            face_size_result, face_size_info = self._face_size_verify(img, face_info)
            if face_size_result == 'small':
                result["small_face"] = True
            elif face_size_result == 'big':
                result["big_face"] = True
            draw_info["face_size_info"] = face_size_info
            
            headpose_result, headpose_info = self._headpose_verify(img, face_info)
            result["headpose"] = headpose_result
            draw_info["headpose_info"] = headpose_info
            
            position_result, position_info = self._position_verify(face_info)
            result['position'] = position_result
            draw_info['position_info'] = str(position_info)
            
            if not headpose_result: # 因為側臉會讓口罩偵測失準，只有正對鏡頭才做口罩偵測
                mask_result, mask_info = self._mask_verify(img, face_info)
                draw_info["mask_info"] = mask_info
                if with_mask==True and mask_result==False:
                    result["put_on_mask"] = True
                elif with_mask==False and mask_result==True:
                    result["put_off_mask"] = True

            img = draw_frame(img, result, draw_info)
        
        return result, img
    
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
            return False, result
        
        return True, result

    def _mask_verify(self, img, face_info):
        x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
        face = img[y:y+h, x:x+w]
        mask, withoutMask = self.detectors['mask'].detect(face)
        
        if mask > 0.7:
            return True, f"Mask : {mask*100:.1f}%"
        else:
            return False, f"Without Mask : {withoutMask*100:.1f}%"

    def _position_verify(self, face_info):
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
        
        if iou < self.config['iou_limit']:
            return True, int(iou*100)
        else:
            return False, int(iou*100)