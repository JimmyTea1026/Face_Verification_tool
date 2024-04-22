import os 
import cv2
import numpy as np
from codes.verificator import Verificator

def draw_frame(frame, face_info):
    if face_info is not None:
        cv2.rectangle(frame, (face_info['x'], face_info['y']), (face_info['x'] + face_info['w'], face_info['y'] + face_info['h']), (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_eye'], 2, (255, 0, 0), 2)
        cv2.circle(frame, face_info['right_eye'], 2, (0, 0, 255), 2)
        cv2.circle(frame, face_info['nose'], 2, (0, 255, 0), 2)
        cv2.circle(frame, face_info['left_mouth'], 2, (255, 0, 255), 2)
        cv2.circle(frame, face_info['right_mouth'], 2, (0, 255, 255), 2)
        cv2.putText(frame, f"conf : {face_info['confidence']:.3f}", (face_info['x']-10, face_info['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
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
        results, face_info = verificator.verify(img)    
        img = draw_frame(img, face_info)
        cv2.imshow('frame', img)
        cv2.waitKey(1)