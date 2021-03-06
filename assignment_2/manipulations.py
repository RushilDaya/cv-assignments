import numpy as np 
import cv2
import math

#_____________HELPER FUNCTIONS _________________________________________________

def _GET_KERNEL_SIZE(minSize, maxSize, percentComplete):

    if minSize % 2 == 0 or maxSize % 2 == 0:
        raise Exception('Min and Max Kernel sizes should be odd numbers')

    tempValue = int(0.01*percentComplete*(maxSize-minSize) + minSize)
    
    if tempValue % 2 == 0:
        tempValue = tempValue - 1

    return min(tempValue, maxSize)

def _CANNY_GET_LOWER(lowerBound, difference, percentComplete):

    maxLower = 255 - difference

    tempValue = int(0.01*percentComplete*(maxLower-lowerBound)+lowerBound)
    return min(tempValue, maxLower)

#_____________MANIPULATION FUNCTIONS __________________________________________    

def sobel(frame, argsObj={}):
    # perform simple horizontal and vertical 
    # sobel filters

    def _numeric_scaling(sobelFrame):
        # abs allows us to view both l/r or t/b edges
        # but we loose the information of the direction of the gradient
        # not important here, but it can be used in other
        # techniques such as the canny edge detection
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

def canny(frame, argsObj ={}):

    lower_bound = argsObj['lower_threshold_start']
    difference =  argsObj['threshold_gap']
    percent_complete = argsObj['percent_complete']

    LOWER_THRESHOLD = _CANNY_GET_LOWER(lower_bound, difference, percent_complete)
    UPPER_THRESHOLD = LOWER_THRESHOLD + difference

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cannyEdges = cv2.Canny(frameGray,LOWER_THRESHOLD, UPPER_THRESHOLD)

    cannyEdges = cv2.cvtColor(cannyEdges, cv2.COLOR_GRAY2BGR)
    cv2.putText(cannyEdges,'CANNY EDGE DETECTOR (lower threshold '+str(LOWER_THRESHOLD) +')( upper threshold '+str(UPPER_THRESHOLD)+')',
                 (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
    return cannyEdges

def houghLine(frame, argsObj={}):
    # code adapted from opencv docs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 200)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    try:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    except:
        pass    

    return frame

def houghCircle(frame, argsObj={}):
    # pretty finicky algorithm 
    # dp: the resolution downgrade factor between hough space and actual
    # minDist: how close can 2 circle centers be
    # param1: the upper threshold of the internal canny filter (lower bound is half)
    # param2: accumulator threshold (smaller -> means more circles detected)
    # minRadius:
    # maxRadius:
    frame = cv2.medianBlur(frame, 5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,dp=20,minDist=50,
                               param1=150,param2=350,minRadius=0,maxRadius=80)
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    except:
        pass
    return frame
    

EFFECT_2_FUNCTION = {
    'sobel': sobel,
    'canny': canny,
    'hough-line':houghLine,
    'hough-circle': houghCircle
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
