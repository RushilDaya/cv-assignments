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