# App : Face verification 

## How to use
**input_form** : (json)
1. id : str
2. with_mask : boolean
3. img_path : str

**output_form** : (json)

1. `id` : str
2. `result` : list - 錯誤狀況的index

    0. "put_off_mask": boolean -  True代表需要拿下口罩
    1. "put_on_mask": boolean -   True代表需要戴上口罩
    2. "many_face": boolean -     True代表偵測到多於一個人臉
    3. "small_face": boolean -    臉太小，True代表太小
    4. "big_face": boolean -      臉太大，True代表太大
    5. "headpose": boolean -      臉有無正對鏡頭，True代表沒有正對鏡頭
    6. "no_face": boolean -       沒有偵測到人臉，True代表沒有偵測到人臉
    7. "position": boolean -      臉有無處在畫面中央，True代表沒有處在畫面中央

**error code** : (json)
1. "Config file not found" - Cannot find the config file
2. "Image file not found" - Cannot find the image file

## Config
`path : ./config/verify.json`
*可以邊使用邊更改config參數，每次輸入新圖片時都會重新讀取config*

Default : 
1. "img_size": [1920, 1080] - **輸入影像大小**
2. "face_size_min": 0.03 - **人臉佔畫面的最小比例**
3. "face_size_max": 0.15 - **人臉佔畫面的最大比例**
---
4. "pitch_limit": 15
5. "yaw_limit": 15
6. "roll_limit": 10
**頭部旋轉的角度限制**
---
7. "mask_limit" : 0.6
**判斷Mask的機率需大於閥值才判斷是有戴口罩**
--- 
8. "valid_area_x": 710
9. "valid_area_y": 290
10. "valid_area_h": 500
11.  "valid_area_w": 500
12.  "iou_limit": 0.8
**在1920x1080畫面中央框出500x500的區域，目標人臉必須與此區域交集大於 `iou_limit`**
---
13.  "save_img": "True"
14.  "save_path": "./result.jpg"
**Debug用，開啟後會在當下目錄存下輸入影像的各種資訊，用於調整參數**

---
## exe generate
1. `pip install pyinstaller`
2. cd 至 main.py所在目錄
3. `pyinstaller -F main.py`
4. 執行後會生成 dist, build 兩個資料夾，exe檔在dist內，將main.exe放到與main.py相同目錄下
---
## Note
- 當臉沒有正對鏡頭時，不會偵測口罩。 (臉歪斜時，口罩偵測效果差)
- ![image](https://hackmd.io/_uploads/HySIT2nbR.png)
使用時可能會有 onnx 的stderr跳出，需要特別處理

