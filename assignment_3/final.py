# this is a step by step file which - 
# produces the serial movie
# assumes the following:
#   1 - the computeEigenFaces.py has been run
#   2 - classification.py has been run

import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import math
import pickle
import os
import copy
import imutils

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 420
FRAME_RATE = 30
VIDEO_NAME = 'output.avi'

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
outputVideo = cv2.VideoWriter(VIDEO_NAME, fourcc, FRAME_RATE, (SCREEN_HEIGHT,SCREEN_WIDTH),1)

def _displayFrame(frame,time=1,frame_rate=FRAME_RATE,fileStream=None, is_gray=True):
    num_frames = int(time*frame_rate)
    frame_d = np.asarray(frame, dtype=np.uint8)
    if is_gray:
        frame_d = cv2.cvtColor(frame_d, cv2.COLOR_GRAY2BGR)
    for i in range(num_frames):
        cv2.imshow('img',frame_d)
        cv2.waitKey(int(1000/frame_rate))
        if fileStream !=None:
            fileStream.write(frame_d)
    return True

def _reshapeFrame(frame, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
    width = int(width)
    height = int(height)
    resized = copy.deepcopy(frame)
    resized = cv2.resize(resized, (height, width), interpolation = cv2.INTER_CUBIC)
    return resized

 
# A - Show some of the raw faces
# -----------------------------------------------------------------
from mixins import getRawImages

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)

text = "Using Raw images of \n Morgan Freeman and \n Conan O'Brian"
y0, dy = 50,30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(displayGrid, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150)

_displayFrame(displayGrid, time=0.2,fileStream=None)

RawImages = getRawImages(10, sample_type='RANDOM', imFormat='GRAYSCALE')
for rawImage in RawImages:
    rawImage = imutils.resize(rawImage, width=min(SCREEN_WIDTH, rawImage.shape[1]))
    rawImage = imutils.resize(rawImage, height=min(SCREEN_HEIGHT, rawImage.shape[0]))
    imgH,imgW = rawImage.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=rawImage
    _displayFrame(displayGrid, time=0.01,fileStream=outputVideo)

# B - explain we use haar cascades to find faces

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)

text = "Faces Extracted from images by \n running  Haar-cascade \n on the raw images"
y0, dy = 50,30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(displayGrid, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150)

_displayFrame(displayGrid, time=0.01,fileStream=None)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)

text = "rejected images with 0 \n or more than 1 \n face detected \n (definite errors)"
y0, dy = 50,30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(displayGrid, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150)

_displayFrame(displayGrid, time=0.05,fileStream=None)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "resize extracted faces \n to 50x50 squares. \n randomly split into \n training and test sets "
y0, dy = 50,30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(displayGrid, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150)

_displayFrame(displayGrid, time=0.05,fileStream=None)

# C - after splitting randomly mix up into training and test set
from mixins import getTrainingImages

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = ""
y0, dy = 50,30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(displayGrid, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150)

_displayFrame(displayGrid, time=0.05,fileStream=None)

trainImages = getTrainingImages()
for image in trainImages:
    image_shaped = imutils.resize(image, height=SCREEN_HEIGHT)
    imgH,imgW = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=image_shaped
    _displayFrame(displayGrid, time=0.005,fileStream=outputVideo)


# D - Show the mean face
from mixins import getMeanFace
from mixins import getEigenFaces

meanFace = getMeanFace()
image_shaped = imutils.resize(meanFace, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=0.5,fileStream=outputVideo)


# E - show some of the eigen-faces after PCA

eigenFaces = getEigenFaces(25)

for image in eigenFaces:
    image_shaped = imutils.resize(image, height=SCREEN_HEIGHT)
    imgH,imgW,_ = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
    displayGrid[0:imgH,0:imgW,:]=image_shaped
    _displayFrame(displayGrid, time=0.01,fileStream=outputVideo, is_gray=False)


# F - show the size of the eigenvalues (indicates set variance)
from mixins import plotEigenValues

eigenValueGraph = plotEigenValues()
image_shaped = imutils.resize(eigenValueGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=5,fileStream=outputVideo, is_gray=False)

# G - show the plot of generalized reconstruction error as a means 
#     to determine the number of PCA components we need

# H - try somehow to show a clustering of faces along eigen axes

# I - take a few faces and draw them with more and fewer eigenfaces

# J - explain briefly the 3 classification methods

# J1 - we must show the new faces in terms of their PCA somehow ( show its the same)

# K - show the accuracy face and accuracy numbers ( for knn show nearest faces)

# L - repeat on more difficult ( non- representitive faces )



outputVideo.release()