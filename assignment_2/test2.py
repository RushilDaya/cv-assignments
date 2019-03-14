# attempting here to determine if we can get a grayscale plot showing intensitity of
# were the dog is based on the SIFT method

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import inspect

MATCH_POINTS = 10
SIFT = cv2.xfeatures2d.SIFT_create()


## process the base image of the dog
####################################################
OBJECT = cv2.imread('dogHead.jpg')
OBJECT_GRAY = cv2.cvtColor(OBJECT, cv2.COLOR_BGR2GRAY)
OBJ_KEYS, OBJ_DESCRIPTORS = SIFT.detectAndCompute(OBJECT_GRAY, None)


## process the image of the dog in the grass
####################################################
SCENE = cv2.imread('dogInGrass.jpg')
SCENE_GRAY = cv2.cvtColor(SCENE, cv2.COLOR_BGR2GRAY)
SCENE_KEYS = SIFT.detect(SCENE_GRAY,None)

## try first looking at a single scale
##################################################
OBJECT_HEIGHT, OBJECT_WIDTH = OBJECT_GRAY.shape
SCENE_HEIGHT, SCENE_WIDTH = SCENE_GRAY.shape

end_horizontal = SCENE_WIDTH - OBJECT_WIDTH
end_vertical = SCENE_HEIGHT - OBJECT_HEIGHT

scan_center_horizontal = int(OBJECT_WIDTH/2)
scan_center_vertical =   int(OBJECT_HEIGHT/2)


def keyInFrame(pt, x_min, x_max, y_min, y_max):
    x,y = pt 
    if x > x_min and x < x_max and y > y_min and y > y_max:
        return True
    else:
        return False

RES_DROP = 30
KNN_SIZE = 2
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

intensityMatrix = np.zeros(( int(end_vertical/RES_DROP), int(end_horizontal/RES_DROP),1), dtype=int)

for x_idx in range(int(end_horizontal/RES_DROP)):
    print(x_idx)
    for y_idx in range(int(end_vertical/RES_DROP)):
       x = x_idx*RES_DROP + 0.5*scan_center_horizontal
       y = y_idx*RES_DROP + 0.5*scan_center_vertical


       SCENE_KEYS_FILTERED = [kp for kp in SCENE_KEYS if keyInFrame(kp.pt, x-0.5*scan_center_horizontal, x+0.5*scan_center_horizontal, y-0.5*scan_center_vertical, y+0.5*scan_center_vertical ) ]
       (_,SCENE_DISCRIPTORS) = SIFT.compute(SCENE_GRAY,SCENE_KEYS_FILTERED)

       if SCENE_DISCRIPTORS is None:
           intensityMatrix[y_idx,x_idx] = 0
       elif len(SCENE_DISCRIPTORS) <= KNN_SIZE:
           intensityMatrix[y_idx, x_idx] = 0
       else:
           matches = flann.knnMatch(OBJ_DESCRIPTORS, SCENE_DISCRIPTORS, k=2)
           good = []
           for m,n in matches:
            if m.distance < 0.7*n.distance:
               good.append(m)
           intensityMatrix[y_idx,x_idx] = len(SCENE_DISCRIPTORS)


maxNum = np.max(intensityMatrix)
intensityMatrix = (1/maxNum)*intensityMatrix
#intensityMatrix = intensityMatrix.astype(int)
cv2.imshow('o',intensityMatrix)
cv2.waitKey(10000)
