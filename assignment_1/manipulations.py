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
    mask[mask>70] = 1

    frame[:,:,0] = blue*mask
    frame[:,:,1] = green*mask
    frame[:,:,2] = red*mask

    cv2.putText(frame,"GRAB BY RGB", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return frame

def grabHSV(frame, argsObj={}):
    # use hsv space to grab the object
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hue = frame[:,:,0]
    saturation = frame[:,:,1]
    value = frame[:,:,2]

    mask = value
    mask[mask>185]=0
    mask[mask<165]=0


    final = mask#cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return final

EFFECT_2_FUNCTION = {
    'grayscale':convert2gray,
    'grayscale-smoothing':grayScaleSmoothing,
    'grayscale-edge-preserve':grayScaleEdgePreserve,
    'color': none,
    'grab-object-rgb': grabRGB,
    'grab-object-hsv': grabHSV,
    'grab-object-morph': none,
    'creative': none,
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
