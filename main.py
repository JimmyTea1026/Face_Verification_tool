import math
import os 
import cv2
import numpy as np
import sys
import json
from codes.verificator import Verificator

def draw_frame(frame, results, draw_info):
    if draw_info is not None:
        face_info = draw_info['face_info']
        headpose_info = draw_info['headpose_info']
        face_size_info = draw_info['face_size_info']
        position_info = draw_info['position_info']
        mask_info = draw_info['mask_info']
        #------------------------------------
        face_size, mask, headpose = results['face_size'], results['with_mask'], results['headpose']
        cv2.putText(frame, f"size:{face_size} / with_mask:{mask} / headpose:{headpose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
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

def realtime_test(verificator):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1920, 1080))
        results, draw_info = verificator.verify(frame)  
        img = draw_frame(frame, results, draw_info)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

def detect(verificator, config, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1920, 1080))
    results, draw_info = verificator.verify(img)
    if config['output_image']:
        img = draw_frame(img, results, draw_info)
        cv2.imwrite('./result.jpg', img)
        
    json_results = json.dumps(results)
    return json_results

if __name__ == "__main__":
    print("Activating")
    config_path='./config.json'
    
    if not os.path.isfile(config_path): 
        sys.stderr.write('config file not found\n')
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
        verificator = Verificator(config)
        # realtime_test(verificator)
        for line in sys.stdin:
            if line.strip() == 'exit':
                break
            img_path = line.strip()
            json_results = detect(verificator, config, img_path)
            print(json_results, file=sys.stdout)
            sys.stdout.flush()