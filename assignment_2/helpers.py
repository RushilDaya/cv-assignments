import numpy as np
import cv2

def numericScaling(frame):
    frame = np.abs(frame)
    mini = np.min(frame)
    maxi = np.max(frame)
    scaled = (1/(maxi-mini))*(frame - mini)
    scaled = np.uint8(scaled*255)
    return scaled

def cannyLowerBound(lowerBound, difference, percentComplete=50):
    maxLower = 255 - difference
    tempValue = int(0.01*percentComplete*(maxLower-lowerBound)+lowerBound)
    return min(tempValue, maxLower)

def siftBoxing(item, scene):
    sift = cv2.xfeatures2d.SIFT_create()

    itemGray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    itemKeyPoints, itemDescriptors = sift.detectAndCompute(itemGray,None)

    sceneGray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sceneKeyPoints, sceneDescriptors = sift.detectAndCompute(sceneGray, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(itemDescriptors, sceneDescriptors, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>10:
        src_pts = np.float32([ itemKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ sceneKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = itemGray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        sceneBlocked = cv2.polylines(scene,[np.int32(dst)],True,(255,70,99),5)
        
    return sceneBlocked

#------------------------------------------
def siftGreyScale(template, scene):
    return scene