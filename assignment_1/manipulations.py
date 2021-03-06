import numpy as np 
import cv2
import math


def convert2gray(frame, argsObj={}):
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"GRAYSCALE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return final

def none(frame, argsObj={}):
    cv2.putText(frame,"COLOR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    return frame

def grayScaleSmoothing(frame, argsObj={}):
    # perform a changing gaussian filter
    KERNEL_SIZE = argsObj['kernel_size']
    VARIANCE_START_VALUE = argsObj['variance_start_value']
    VARIANCE_END_VALUE = argsObj['variance_end_value']
    PERCENT_COMPLETE = argsObj['percent_complete']

    # determine the variance based on the percentage complete
    VARIANCE = (VARIANCE_END_VALUE - VARIANCE_START_VALUE)*(PERCENT_COMPLETE/100)-VARIANCE_START_VALUE
    
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(temp,(KERNEL_SIZE,KERNEL_SIZE),VARIANCE)
    final = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"GAUSSIAN FILTER", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return final

def grayScaleEdgePreserve(frame, argsObj={}):
    # perform a smoothing operation which preserves edges
    MIN_APPLICATIONS = argsObj['min_applications']
    MAX_APPLICATIONS = argsObj['max_applications']
    PERCENT_COMPLETE = argsObj['percent_complete']

    NUM_APPLICATIONS = math.ceil((MAX_APPLICATIONS - MIN_APPLICATIONS)*(PERCENT_COMPLETE/100)+MIN_APPLICATIONS)
    
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(NUM_APPLICATIONS):
        print(i)
        temp = cv2.bilateralFilter(temp, 5, 75,75)

    final = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"BILATERAL FILTER", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return final

def grabRGB(frame, argsObj={}):

    # extract just the red channel 
    blue = frame[:,:,0]
    green = frame[:,:,1]
    red = frame[:,:,2]

    mask = red - green # color combination and thereshold determined empirically
    mask[mask>200] = 0
    mask[mask<70] = 0
    mask[mask>=70] = 1

    frame[:,:,0] = blue*mask
    frame[:,:,1] = green*mask
    frame[:,:,2] = red*mask

    cv2.putText(frame,"GRAB BY RGB", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def grabMorp(frame, argsObj={}):

    # extract just the red channel 
    blue = frame[:,:,0]
    green = frame[:,:,1]
    red = frame[:,:,2]

    mask = red - green # color combination and thereshold determined empirically
    mask[mask>200] = 0
    mask[mask<70] = 0
    mask[mask>=70] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(18,18))
    mask_2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    frame[:,:,0] = mask*255
    frame[:,:,1] = mask_2*255
    frame[:,:,2] = 0

    cv2.putText(frame,"GRAB BY RGB WITH CLOSING", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def grabHSV(frame, argsObj={}):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hue = np.copy(hsv[:,:,0]) # hue tracking was found to be the most realiable
    hue[hue > 179 ] = 0
    hue[hue < 177 ] = 0
    hue[hue > 0] = 1
    
    
    frame[:,:,0] = frame[:,:,0]*hue
    frame[:,:,1] = frame[:,:,1]*hue
    frame[:,:,2] = frame[:,:,2]*hue

    cv2.putText(frame,"GRAB BY HSV", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def detectFaces(frame, argsObj={}):
    # haar cascade based face detection
    # based on tutorial from https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html

    face_cascade = cv2.CascadeClassifier('haar_face_cascade.xml')

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
    
    cv2.putText(frame,"FACE DETECTION", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def facialBlurring(frame, argsObj={}):
    # detection based on the detect faces routine
    # additional effect of blurring

    face_cascade = cv2.CascadeClassifier('haar_face_cascade.xml')
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)

    BlurredFrame = cv2.GaussianBlur(frame,(51,51),51)

    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:]=BlurredFrame[y:y+h,x:x+w,:]

    cv2.putText(frame,"BLUR FACE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def laplacianFiltering(frame, argsObj={}):
    # laplacian filter with pre- and post- smoothing to improve
    # the filter
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.GaussianBlur(frame[:,:,2],(21,21),21)
    laplacian = cv2.Laplacian(grayImage,cv2.CV_64F)
    
    laplacian = cv2.blur(laplacian, (11,11))
    mini = np.min(laplacian)
    maxi = np.max(laplacian)
    laplacian = (1/(maxi-mini))*(laplacian - mini)
    laplacian[laplacian<0.65] = 0
    laplacian=laplacian*255
    laplacian = np.uint8(laplacian)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    cv2.putText(laplacian,"SMOOTH LAPLACIAN", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    
    return laplacian


EFFECT_2_FUNCTION = {
    'grayscale':convert2gray,
    'grayscale-smoothing':grayScaleSmoothing,
    'grayscale-edge-preserve':grayScaleEdgePreserve,
    'color': none,
    'grab-object-rgb': grabRGB,
    'grab-object-hsv': grabHSV,
    'grab-object-rgb-morp': grabMorp,
    'creative': none,
    'face-detect': detectFaces,
    'face-blur': facialBlurring,
    'laplace-filter': laplacianFiltering,
    'none':none
}

def getMethod(frame_number, frame_rate, effects):
    keySet = list(effects.keys())
    timePoint = frame_number/frame_rate

    effectObj = effects[keySet[-1]]
    for i in range(len(keySet)):
        if timePoint < keySet[i]:
            effectObj = effects[keySet[i-1]]
            break

    # calculate where along in the effect we are
    timeInEffect = timePoint - keySet[i-1] 
    effectObj['percent_complete'] = 100*timeInEffect/effectObj['len']

    return [EFFECT_2_FUNCTION[effectObj['name']], effectObj]
