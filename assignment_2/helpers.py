import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt

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
        sceneCopy = copy.deepcopy(scene)
        sceneBlocked = cv2.polylines(sceneCopy,[np.int32(dst)],True,(255,70,99),5)
        
    return sceneBlocked

def keyInFrame(pt, x_min, x_max, y_min, y_max):
    x,y = pt 
    if x > x_min and x < x_max and y > y_min and y < y_max:
        return True
    else:
        return False

#------------------------------------------
def siftGreyScale(template, sceneX):
    RESOLUTION_REDUCTION_FACTOR = 10
    KNN_SIZE = 2
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    scene = copy.deepcopy(sceneX)
    sift = cv2.xfeatures2d.SIFT_create()

    itemGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    itemKeyPoints, itemDescriptors = sift.detectAndCompute(itemGray,None)

    sceneGray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sceneKeyPoints = sift.detect(sceneGray, None)

    itemHeight, itemWidth = itemGray.shape
    sceneHeight, sceneWidth = sceneGray.shape

    endHorizontal = int(sceneWidth - itemWidth/2)
    endVertical = int(sceneHeight - itemHeight/2)

    scanCenterHorizontal = int(itemWidth/2)
    scanCenterVertical = int(itemHeight/2)



    intensityMatrix = np.zeros(( int(endVertical/RESOLUTION_REDUCTION_FACTOR), int(endHorizontal/RESOLUTION_REDUCTION_FACTOR)), dtype=int)

    for x_idx in range(int(endHorizontal/RESOLUTION_REDUCTION_FACTOR)):
        print (x_idx)
        print (int(endHorizontal/RESOLUTION_REDUCTION_FACTOR))
        print ('---')
        for y_idx in range(int(endVertical/RESOLUTION_REDUCTION_FACTOR)):
            x = x_idx*RESOLUTION_REDUCTION_FACTOR + 0.5*scanCenterHorizontal
            y = y_idx*RESOLUTION_REDUCTION_FACTOR + 0.5*scanCenterVertical

            sceneKeysFiltered = [kp for kp in sceneKeyPoints if keyInFrame(kp.pt, x-0.5*scanCenterHorizontal, x+0.5*scanCenterHorizontal, y-0.5*scanCenterVertical, y+0.5*scanCenterVertical ) ]
            (_,sceneDescriptors) = sift.compute(sceneGray, sceneKeysFiltered)
            
            if sceneDescriptors is None:
                intensityMatrix[y_idx, x_idx] = 0
            elif len(sceneDescriptors) <= KNN_SIZE:
                intensityMatrix[y_idx, x_idx] = 0
            else:
                matches = flann.knnMatch(itemDescriptors, sceneDescriptors, k=KNN_SIZE)
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                intensityMatrix[y_idx,x_idx] = len(good)

    maxNum = np.max(intensityMatrix)
    intensityMatrix = (1/maxNum)*intensityMatrix
    #intensityMatrix = np.flip(intensityMatrix) 

    # -- need to take the binned data and make it into an image again
    newScene = np.zeros((sceneHeight,sceneWidth,1), dtype=int)
    print(endVertical)
    print(endHorizontal)
    for yPixel in range(sceneHeight):
        for xPixel in range(sceneWidth):
            if yPixel < scanCenterVertical:
                pass 
            elif yPixel > endVertical:
                pass
            elif xPixel < scanCenterHorizontal:
                pass 
            elif xPixel > endHorizontal:
                pass
            else:
                mapX = int(xPixel/RESOLUTION_REDUCTION_FACTOR)
                mapY = int(yPixel/RESOLUTION_REDUCTION_FACTOR)
                try:
                    newScene[yPixel,xPixel]=intensityMatrix[mapY,mapX]*255
                except:
                    pass


    #plt.imshow(newScene)
    #plt.show()
    newScene = newScene.astype('uint8')
    newScene = cv2.cvtColor(newScene, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('o',newScene)
    #cv2.waitKey(20000)               
    return newScene