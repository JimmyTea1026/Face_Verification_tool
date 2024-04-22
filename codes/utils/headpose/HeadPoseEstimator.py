import cv2
import numpy as np
import math
import onnxruntime as ort
import onnx

def squareCrop(image, face, scaled=0.3, targetSize=(112,112)):
    x, y, w, h = face[:4].astype(np.int32)

    m = max(w, h)

    new_side = int(m + m * scaled)
    half_side = new_side //2

    cx = x + w//2
    cy = y + h//2

    cy_start = int(np.clip(cy - half_side, 0, image.shape[0]))
    cy_end = int(np.clip(cy + half_side, 0, image.shape[0]))
    cx_start = int(np.clip(cx - half_side, 0, image.shape[1]))
    cx_end = int(np.clip(cx + half_side, 0, image.shape[1]))

    crop_image = image[cy_start:cy_end, cx_start:cx_end, :]

    ch, cw, _ = crop_image.shape
    ms = max(ch, cw)

    # 計算補充的邊界大小
    yPadding = (ms - crop_image.shape[0]) // 2
    xPadding = (ms - crop_image.shape[1]) // 2

    # 使用copyMakeBorder進行補充
    padded_image = cv2.copyMakeBorder(crop_image, yPadding, yPadding, xPadding, xPadding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    squareImage = cv2.resize(padded_image,targetSize)

    return squareImage

class HeadPoseEstimator:
    def __init__(self, modelPath):
        self.initParm(modelPath)

    def initParm(self, modelPath):
        # self.estimator = cv2.dnn.readNetFromONNX(modelPath)
        self.model = onnx.load(modelPath)
        self.input_size = (224,224)

        self.mean= np.array([0.485, 0.456, 0.406], dtype=np.float64)
        self.std=np.array([0.229, 0.224, 0.225], dtype=np.float64)

    def preprocess(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, self.input_size)
        
        img = img / 255.0

        img = (img - self.mean)/self.std

        img = np.transpose(img, (2, 0, 1))

        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        return input_data
    
    def preprocess_cv(self, img, input_size=(224,224), 
                   mean=np.array([0.485, 0.456, 0.406], dtype=np.float64), 
                   std=np.array([0.229, 0.224, 0.225], dtype=np.float64)):
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, input_size)
        
        img = img / 255.0

        img = img.astype(np.float32)

        r,g,b = cv2.split(img)

        r = cv2.subtract(r, 0.485)
        r = cv2.divide(r, 0.229)        

        g = cv2.subtract(g, 0.456)
        g = cv2.divide(g, 0.224) 

        b = cv2.subtract(b, 0.406)
        b = cv2.divide(b, 0.225)

        img = cv2.merge([r,g,b])      

        input_data = cv2.dnn.blobFromImage(img, swapRB=False, ddepth= 5)

        return input_data

    def predict(self, input_data):
        session = ort.InferenceSession(self.model.SerializeToString())
        outputs = session.run(None, {'input': input_data})

        output_euler = self.compute_euler_angles_from_rotation_matrices(outputs[0])

        return output_euler
    
    def compute_euler_angles_from_rotation_matrices(self, rotation_matrices):
        R = rotation_matrices[0].copy()

        sy = math.sqrt(R[0][0]**2 + R[1][0] ** 2)
        singular = 1.0 if sy < 1e-6 else 0.0 # false == 0 , true == 1

        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])

        xs = math.atan2(-R[1][2], R[1][1])
        ys = math.atan2(-R[2][0], sy)
        zs = 0.0

        out_euler = np.zeros(3)
        out_euler[0] = x * (1-singular) + xs * singular
        out_euler[1] = y * (1-singular) + ys * singular
        out_euler[2] = z * (1-singular) + zs * singular

        out_euler = out_euler * 180 / np.pi

        return out_euler
    
    def run(self, image, box):
        '''
        Input parameter : origin image + bbox of face
        '''
        box = np.array(box)
        squareImage = squareCrop(image, box, targetSize=self.input_size, scaled=0.25)

        input_data = self.preprocess_cv(squareImage)

        output = self.predict(input_data)

        # pitch, yaw, roll = output_euler
        return output