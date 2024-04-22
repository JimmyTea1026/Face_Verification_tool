import math
import os 
import cv2
import numpy as np
from codes.verificator import Verificator

def draw_frame(frame, draw_info):
    if draw_info is not None:
        face_info = draw_info['face_info']
        headpose_info = draw_info['headpose_info']
        face_size_info = draw_info['face_size_info']
        #------------------------------------
        cv2.rectangle(frame, (face_info['x'], face_info['y']), (face_info['x'] + face_info['w'], face_info['y'] + face_info['h']), (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_eye'], 2, (255, 0, 0), 2)
        cv2.circle(frame, face_info['right_eye'], 2, (0, 0, 255), 2)
        cv2.circle(frame, face_info['nose'], 2, (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_mouth'], 2, (255, 0, 255), 2)
        cv2.circle(frame, face_info['right_mouth'], 2, (0, 255, 255), 2)
        # cv2.putText(frame, f"conf : {face_info['confidence']:.3f}", (face_info['x']-10, face_info['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        #------------------------------------
        cv2.putText(frame, f"size : {int(face_size_info)}%", (face_info['x']-10, face_info['y'] + face_info['h'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
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

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)
        
    return frame

if __name__ == "__main__":
    img_size = (640, 480) 
    verificator = Verificator(img_size, config_path='./config.json')

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, img_size)
        results, draw_info = verificator.verify(img)  
        img = draw_frame(img, draw_info)
        cv2.imshow('frame', img)
        cv2.waitKey(1)