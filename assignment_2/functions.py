import numpy as np
import cv2
import copy
import helpers as hlp

FRAME_RATE = 60.0
RESOLUTION = (1280, 720)
CANNY_LOWER = 10
CANNY_THRESHOLD_GAP = 100 

def openVideo(name):
    captureObj = cv2.VideoCapture(name)
    return captureObj

def createOutput(name):
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    outputObj =  cv2.VideoWriter(name, fourcc, FRAME_RATE, RESOLUTION, 1)
    return outputObj

def openImage(name):
    return cv2.imread(name)


def baseVideo(inputStream, outputStream, time):

    numberFrames = int(time*FRAME_RATE)
    frameIdx=0
    while inputStream.isOpened() and frameIdx < numberFrames:
        ret, frame = inputStream.read()
        if ret == True:
            outputStream.write(frame)
        else:
            break
        frameIdx +=1
    return True

def sobelVideo(inputStream, outputStream, time):
    
    numberFrames = int(time*FRAME_RATE)
    frameIdx=0
    while inputStream.isOpened() and frameIdx < numberFrames:
        ret, frame = inputStream.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel_size = 5
            sobelHorizontal = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=kernel_size)
            sobelHorizontal = hlp.numericScaling(sobelHorizontal)
            sobelVertical = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobelVertical = hlp.numericScaling(sobelVertical)
            frame[:,:,0] = 0
            frame[:,:,1] = sobelHorizontal
            frame[:,:,2] = sobelVertical
            cv2.putText(frame,'SOBEL', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            outputStream.write(frame)
        else:
            break
        frameIdx +=1
    return True 


def cannyVideo(inputStream, outputStream, time):
    numberFrames = int(time*FRAME_RATE)
    frameIdx = 0
    while inputStream.isOpened() and frameIdx < numberFrames:
        ret, frame = inputStream.read()
        if ret == True:
            lowerThreshold = hlp.cannyLowerBound(CANNY_LOWER, CANNY_THRESHOLD_GAP)
            upperThreshold = lowerThreshold + CANNY_THRESHOLD_GAP
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cannyEdges = cv2.Canny(gray, lowerThreshold, upperThreshold)
            cannyEdges = cv2.cvtColor(cannyEdges, cv2.COLOR_GRAY2BGR)
            cv2.putText(cannyEdges,'Canny Edge Detector',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
            outputStream.write(cannyEdges)
        else:
            break
        frameIdx +=1
    return True

def houghCircleVideo(inputStream, outputStream, time):
    numberframes = int(time*FRAME_RATE)
    frameIdx = 0
    while inputStream.isOpened() and frameIdx < numberframes:
        ret, frame = inputStream.read()
        if ret == True:
            frame = cv2.medianBlur(frame, 5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=20,minDist=50,
                                       param1=150, param2=350, minRadius=0, maxRadius=80)
            try:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                    cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            except:
                pass
            cv2.putText(frame,'Hough Circles', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
            outputStream.write(frame)
        else:
            break
        frameIdx +=1
    return True


def doSift(template, scene, outputStream, lengthBox, lengthGreyScale):
    
    sceneBoxed = hlp.siftBoxing(template, scene)
    sceneGreyScale = hlp.siftGreyScale(template, scene)
    
    numberFrames = int(lengthBox*FRAME_RATE)
    frameIdx = 0
    while frameIdx < numberFrames:
        outputStream.write(sceneBoxed)
        frameIdx +=1

    numberFrames = int(lengthGreyScale*FRAME_RATE)
    frameIdx = 0
    while frameIdx < numberFrames:
        outputStream.write(sceneGreyScale)
        frameIdx +=1
    
    return True

def doSiftVideo(template, inputStream, outputStream, length):
    sift = cv2.xfeatures2d.SIFT_create()
    
    itemGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    itemKeyPoints, itemDescriptors = sift.detectAndCompute(itemGray,None)   

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    numberFrames = int(length*FRAME_RATE)
    frameIdx = 0
    while inputStream.isOpened() and frameIdx < numberFrames:
        print(frameIdx, numberFrames)
        ret, frame = inputStream.read()
        if ret == True:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameKeyPoints, frameDescriptors = sift.detectAndCompute(frameGray, None)
            matches = flann.knnMatch(itemDescriptors, frameDescriptors, k=2)
            good =[]
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            if len(good)>10:
                src_pts = np.float32([ itemKeyPoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ frameKeyPoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = itemGray.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                frameCopy = copy.deepcopy(frame)
                frame = cv2.polylines(frameCopy,[np.int32(dst)],True,(255,70,99),5)
            outputStream.write(frame)
        else:
            pass
        frameIdx +=1        
        
    return True
 

def combine(videos, outputStream):
    for video in videos:
        while video.isOpened():
            ret, frame = video.read()
            if ret == True:
                outputStream.write(frame)
            else:
                break
    return True
