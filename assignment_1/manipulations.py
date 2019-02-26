import numpy as np 
import cv2


def convert2gray(frame, argsObj={}):
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"GRAYSCALE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    print(argsObj['percent_complete'])
    return final

def none(frame, argsObj={}):
    cv2.putText(frame,"COLOR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    return frame

def grayScaleSmoothing(frame, argsObj={}):
    # perform a non-edge preserving filter
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered = cv2.GaussianBlur(temp,(5,5),0)

    final = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"GAUSSIAN FILTER", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return final

def grayScaleEdgePreserve(frame, argsObj={}):
    # perform a smoothing operation which preserves edges
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(temp,9,75,75 )

    final = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(final,"BILATERAL FILTER", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 150)
    return final

def grabRGB(frame, argsObj={}):
    # attempting to grab the green sponge in the video
    return frame



EFFECT_2_FUNCTION = {
    'grayscale':convert2gray,
    'grayscale-smoothing':grayScaleSmoothing,
    'grayscale-edge-preserve':grayScaleEdgePreserve,
    'color': none,
    'grab-object-rgb': grabRGB,
    'grab-object-hsv': none,
    'grab-object-morph': none,
    'creative': none,
    'none':none
}

def getMethod(frame_number, frame_rate, effects):
    keySet = list(effects.keys())
    timePoint = frame_number/frame_rate

    method = effects[keySet[-1]]
    for i in range(len(keySet)):
        if timePoint < keySet[i]:
            effectObj = effects[keySet[i-1]]
            break

    # calculate where along in the effect we are
    timeInEffect = timePoint - keySet[i-1] 
    effectObj['percent_complete'] = 100*timeInEffect/effectObj['len']

    return [EFFECT_2_FUNCTION[effectObj['name']], effectObj]
