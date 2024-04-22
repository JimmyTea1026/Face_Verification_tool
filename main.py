import os 
import cv2
import numpy as np
from codes.verificator import Verificator


def verification(img, detetors):
    pass


if __name__ == "__main__":
    img_size = (1080, 1920) 
    verificator = Verificator(img_size, face_ratio=0, headpose_angle=0)

    # Load image
    img = cv2.imread('./test/1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    