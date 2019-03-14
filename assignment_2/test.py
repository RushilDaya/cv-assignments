# The following tutorial was used as a starting point for the assignment as it gives insight in how to read video files
# https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html?fbclid=IwAR0LYIh0gPbtVyeMiD6eRXfwCKR4eDNXF_ANPxrEi_2Ewioe87GU3GErgxE

import numpy as np 
import cv2
import manipulations as mp
from matplotlib import pyplot as plt


FRAME_RATE = 60.0
RESOLUTION = (1280, 720)
MATCH_POINTS = 10


objImg = cv2.imread('dog.jpeg')
objGray = cv2.cvtColor(objImg, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kpObj, desObj = sift.detectAndCompute(objGray, None)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kpImg, desImg = sift.detectAndCompute(imgGray, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desObj,desImg,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MATCH_POINTS:
        src_pts = np.float32([ kpObj[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpImg[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = objGray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        print(pts)
        imgGray = cv2.fillPoly(imgGray,[np.int32(dst)],0)

    else:
        matchesMask = None

    cv2.imshow('image',imgGray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()