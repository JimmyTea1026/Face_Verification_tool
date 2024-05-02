import os 
import cv2
import sys
import json
from codes.verificator import Verificator

def realtime_test(verificator):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results, drawed_img = verificator.verify(frame, with_mask=True)  
        r = postprocess(results, 0)
        print(json.dumps(r))
        drawed_img = cv2.resize(drawed_img, (640, 480))
        cv2.imshow('frame', drawed_img)
        cv2.waitKey(1)

def img_verify(verificator):
    print(json.dumps("Activated"))
    sys.stdout.flush()
    
    for line in sys.stdin:
        if line.strip() == 'exit':
            sys.exit()

        input_data = json.loads(line.strip())
        _id, img_path, with_mask = input_data['id'], input_data['img_path'], input_data['with_mask']
        
        try:
            if os.path.isfile(img_path):
                verify_results, drawed_img = verificator.verify(img_path, with_mask)
                results = postprocess(verify_results, _id)                
                json_results = json.dumps(results)
                print(json_results)
                sys.stdout.flush()
        except:
            print(json.dumps({"error": "Image file not found"}))
            sys.stdout.flush()

def test(verificator):
    _id = 0
    img_path = "D:\\Compal\\Code\\face verification\\test\\2.jpg"
    with_mask = False
    verify_results, drawed_img = verificator.verify(img_path, with_mask)
    results = postprocess(verify_results, _id)                
    json_results = json.dumps(results)
    print(json_results)
    save_path = "D:\\Compal\\Code\\face verification\\test\\result.jpg"
    cv2.imwrite(save_path, drawed_img)

def postprocess(verify_results, _id):
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
    error_type = {'put_off_mask': 0, 'put_on_mask': 1, 'many_face': 2, 'small_face': 3, 'big_face': 4, 'headpose': 5, 'no_face': 6, 'position': 7}
    results = {'id': None, 'result': []}
    results['id'] = _id
    for key, value in verify_results.items():
        if value == True:
            results['result'].append(error_type[key])
    
    return results

if __name__ == "__main__":
    config_path='./config/verify_config.json'
    
    if not os.path.isfile(config_path): 
        msg = json.dumps({"error": "Config file not found"})
        print(msg)
        sys.exit()

    verificator = Verificator(config_path)
    # test(verificator)
    realtime_test(verificator)
    # img_verify(verificator, config)