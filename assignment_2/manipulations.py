import numpy as np 
import cv2
import math



def _GET_KERNEL_SIZE(minSize, maxSize, percentComplete):

    if minSize % 2 == 0 or maxSize % 2 == 0:
        raise Exception('Min and Max Kernel sizes should be odd numbers')

    tempValue = int(0.01*percentComplete*(maxSize-minSize) + minSize)
    
    if tempValue % 2 == 0:
        tempValue = tempValue - 1

    return min(tempValue, maxSize)

def sobel(frame, argsObj={}):
    # perform simple horizontal and vertical 
    # sobel filters

    def _numeric_scaling(sobelFrame):
        sobelFrame = np.abs(sobelFrame)
        mini = np.min(sobelFrame)
        maxi = np.max(sobelFrame)
        scaled = (1/(maxi-mini))*(sobelFrame - mini)
        scaled = np.uint8(scaled*255)
        return scaled

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_kernel = argsObj['kernel_start_size']
    max_kernel = argsObj['kernel_end_size']
    percent_complete = argsObj['percent_complete']
    kernel_size = _GET_KERNEL_SIZE(min_kernel,max_kernel,percent_complete)

    # get horizontal sobel
    sobelHorizontal = cv2.Sobel(frameGray,cv2.CV_64F,0,1, ksize=kernel_size)
    sobelHorizontal = _numeric_scaling(sobelHorizontal)
    # get vertical sobel
    sobelVertical = cv2.Sobel(frameGray,cv2.CV_64F,1,0, ksize=kernel_size)
    sobelVertical = _numeric_scaling(sobelVertical)

    # use the green and red layers to show vertical and horizontal edges
    frame[:,:,0] = 0
    frame[:,:,1] = sobelHorizontal
    frame[:,:,2] = sobelVertical

    cv2.putText(frame,'SOBEL (kernel size: '+str(kernel_size)+')', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    return frame


EFFECT_2_FUNCTION = {
    'sobel': sobel
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
    effectObj['percent_complete'] = min(100, 100*timeInEffect/effectObj['len'])

    return [EFFECT_2_FUNCTION[effectObj['name']], effectObj]
