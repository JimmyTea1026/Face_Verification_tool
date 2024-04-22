import os
import cv2
import numpy as np

from utils.FaceDetect.Scrfd import Scrfd
from utils.FaceRecognize.ArcFace import ArcFace
from utils.HeadPose.HeadPoseEstimator import HeadPoseEstimator
from utils.Tool.DataManager import loadDB
from utils.Stream.StreamProvider import Stream
from utils.Draw.FacePainter import FacePainter


# featureDB, nameList = loadDB("featureDB/ArcFaceFeature_all_noMask.tiff", "featureDB/ArcFaceName_all_noMask.txt")

detector = Scrfd(modelPath="weights/scrfd_dm.onnx")
recognizer = ArcFace(modelPath = "weights/arcface.onnx")
estimator = HeadPoseEstimator(modelPath = "weights/headpose_A0_new.onnx")
# painter = FacePainter(nameList)

cameraSize = (1920,1080) # w, h
stream = Stream(src=1, iw=cameraSize[0], ih=cameraSize[1])

###
win_name = "demo"
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
###

while True:
    img_ori = stream.read()
    img_ori = cv2.flip(img_ori, 2)

    img_show = img_ori.copy()

    faces = detector.easyDetect(img_ori)  #output is [n , 15]
    alignList = []
    if faces is not None:
        for index, face in enumerate(faces):
            #############################################
            face_align_image, face_feature = recognizer.convert2feature(img_ori.copy(), face)

            # predictResults = recognizer.compareWithAll_topN(face_feature, featureDB, nameList, N=1)
            # predictName, predictScore = predictResults[0]

            alignList.append(face_align_image)
            #############################################
            output_euler = estimator.run(img_ori.copy(), face) 

           # pitch, yaw, roll = output_euler

            #############################################
            showInfo = dict()
            # showInfo["score"] = predictScore
            # showInfo["name"] = predictName
            showInfo["conf"] = face[-1]
            showInfo["headPose"] = output_euler
            #############################################
            ims_show = painter.drawFace(img_show, face, showInfo, drawPoint=True)

        #############################################
        img_result = np.hstack(alignList)
        img_show[:img_result.shape[0],:img_result.shape[1], :] = img_result

    cv2.imshow(win_name, img_show)

    key = cv2.waitKeyEx(1)

    if key != -1:
        if key == 113:  # q
            break

stream.stop()
cv2.destroyAllWindows()