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
outputVideo = cv2.VideoWriter(VIDEO_NAME, fourcc, FRAME_RATE, (SCREEN_WIDTH,SCREEN_HEIGHT),1)

def _displayFrame(frame,time=1,frame_rate=FRAME_RATE,fileStream=outputVideo, is_gray=True):
    num_frames = int(time*frame_rate)
    frame_d = np.asarray(frame, dtype=np.uint8)
    if is_gray:
        frame_d = cv2.cvtColor(frame_d, cv2.COLOR_GRAY2BGR)
    for i in range(num_frames):
        #cv2.imshow('img',frame_d)
        #cv2.waitKey(int(1000/frame_rate))
        if fileStream !=None:
            fileStream.write(frame_d)
    return True

def _reshapeFrame(frame, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
    width = int(width)
    height = int(height)
    resized = copy.deepcopy(frame)
    resized = cv2.resize(resized, (height, width), interpolation = cv2.INTER_CUBIC)
    return resized

def _write_text(frame, text):
    y0, dy = 50,30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(frame, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 150) 
    return frame

# A - Show some of the raw faces
# -----------------------------------------------------------------
from mixins import getRawImages

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)

text = "Using Raw images of \n Morgan Freeman and \n Conan O'Brian"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

RawImages = getRawImages(10, sample_type='RANDOM', imFormat='GRAYSCALE')
for rawImage in RawImages:
    rawImage = imutils.resize(rawImage, width=min(SCREEN_WIDTH, rawImage.shape[1]))
    rawImage = imutils.resize(rawImage, height=min(SCREEN_HEIGHT, rawImage.shape[0]))
    imgH,imgW = rawImage.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=rawImage
    _displayFrame(displayGrid, time=0.3,fileStream=outputVideo)

# B - explain we use haar cascades to find faces

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)

text = "Faces Extracted from images by \n running  Haar-cascades"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "resize extracted faces \n to 50x50 squares. \n  split into \n training and test sets "
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=2,fileStream=outputVideo)

# C - after splitting randomly mix up into training and test set
from mixins import getTrainingImages

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Example of \n Training Faces"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

trainImages = getTrainingImages()[0:3]+getTrainingImages()[-3:-1]
for image in trainImages:
    image_shaped = imutils.resize(image, height=SCREEN_HEIGHT)
    imgH,imgW = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=image_shaped
    _displayFrame(displayGrid, time=0.3,fileStream=outputVideo)


# D - Show the mean face
from mixins import getMeanFace
from mixins import getEigenFaces

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "First Step to \n generating the eigenfaces \n is getting a mean face \n shown next"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

meanFace = getMeanFace()
image_shaped = imutils.resize(meanFace, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=1,fileStream=outputVideo)


# E - show some of the eigen-faces after PCA

eigenFaces = getEigenFaces(6)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Next Step is running \n pca to get eigenfaces \n shown next are the 6 first \n eigenfaces"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

for image in eigenFaces:
    image_shaped = imutils.resize(image, height=SCREEN_HEIGHT)
    imgH,imgW,_ = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
    displayGrid[0:imgH,0:imgW,:]=image_shaped
    _displayFrame(displayGrid, time=0.7,fileStream=outputVideo, is_gray=False)


# F - show the size of the eigenvalues (indicates set variance)
from mixins import plotEigenValues

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "The Eigenvalues corresponding to \n eigenfaces indict the variance \n of the training data \n in that direction"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=2,fileStream=outputVideo)


eigenValueGraph = plotEigenValues()
image_shaped = imutils.resize(eigenValueGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=False)

# G - show the plot of generalized reconstruction error as a means 
#     to determine the number of PCA components we need
from mixins import plotReconstructionError

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Next we see the reconstruction error \n across the training data as \n we use more and more pca \n components in the reconstruction"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

ReconstructionErrorGraph = plotReconstructionError()
image_shaped = imutils.resize(ReconstructionErrorGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=False)

# H - try somehow to show a clustering of faces along eigen axes
from mixins import visualizeClustering

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Now we look at 2D scatter plots \n along some eigen faces"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Notice it is difficult to \n separate this set with \n only 2 dimensions"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

clusterGraph = visualizeClustering(0,1)
image_shaped = imutils.resize(clusterGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=2,fileStream=outputVideo, is_gray=False)

clusterGraph = visualizeClustering(2,3)
image_shaped = imutils.resize(clusterGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=2,fileStream=outputVideo, is_gray=False)

clusterGraph = visualizeClustering(4,5)
image_shaped = imutils.resize(clusterGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=2,fileStream=outputVideo, is_gray=False)

clusterGraph = visualizeClustering(6,9)
image_shaped = imutils.resize(clusterGraph, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=2,fileStream=outputVideo, is_gray=False)

# I - take a few faces and draw them with more and fewer eigenfaces
from mixins import drawFaceIncrementally

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Here we show the reconstruction \n of faces from incrementally \n more eigenfaces"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

images = drawFaceIncrementally()
for image in images:
    image_shaped = imutils.resize(image, height=int(SCREEN_HEIGHT*0.5))
    imgH,imgW = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=image_shaped
    _displayFrame(displayGrid, time=0.3,fileStream=outputVideo, is_gray=True)

images = drawFaceIncrementally()
for image in images:
    image_shaped = imutils.resize(image, height=int(SCREEN_HEIGHT*0.5))
    imgH,imgW = image_shaped.shape
    displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
    displayGrid[0:imgH,0:imgW]=image_shaped
    _displayFrame(displayGrid, time=0.3,fileStream=outputVideo, is_gray=True)
    
# J - explain briefly the 3 classification methods


# J1 - we must show the new faces in terms of their PCA somehow ( show its the same)
from mixins import getTrainingTestHeatMap

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Now we look at PCA weights for \n training sets and test sets. \n Randomly chosen to have \n similar distributions"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

heatMap = getTrainingTestHeatMap('./data/training_faces/person_1/','./data/test_faces/person_1/')
image_shaped = imutils.resize(heatMap, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=5,fileStream=outputVideo, is_gray=False)

heatMap = getTrainingTestHeatMap('./data/training_faces/person_2/','./data/test_faces/person_2/')
image_shaped = imutils.resize(heatMap, height=SCREEN_HEIGHT)
imgH,imgW,_ = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3),dtype='int')
displayGrid[0:imgH,0:imgW,:]=image_shaped
_displayFrame(displayGrid, time=5,fileStream=outputVideo, is_gray=False)


# K - show the accuracy face and accuracy numbers ( for knn show nearest faces)
from classification import boostedTreePredict, svmPredict, knnClassify
from mixins import compositeImage

test_path_1 = './data/test_faces/person_1/'
test_path_2 = './data/test_faces/person_2/'

test_paths = [test_path_1+item for item in os.listdir(test_path_1)]+[test_path_2+item for item in os.listdir(test_path_2)]

knnPredictions = [knnClassify(item) for item in test_paths]
svmPredictions = [svmPredict(item) for item in test_paths]
boostedTreePredictions = [boostedTreePredict(item) for item in test_paths]

knnPred1 = []
knnPred2 = []
svmPred1 = []
svmPred2 = []
btPred1 = []
btPred2 = []

for i in range(len(test_paths)):
    if knnPredictions[i] == 1:
        knnPred1 +=[test_paths[i]]
    else:
        knnPred2 +=[test_paths[i]]

    if svmPredictions[i] == 1:
        svmPred1 +=[test_paths[i]]
    else:
        svmPred2 +=[test_paths[i]]

    if boostedTreePredictions[i] ==1:
        btPred1 +=[test_paths[i]]
    else:
        btPred2 +=[test_paths[i]]


knnPerson1Composite = compositeImage(knnPred1)
knnPerson2Composite = compositeImage(knnPred2)

svmPerson1Composite = compositeImage(svmPred1)
svmPerson2Composite = compositeImage(svmPred2)
btPerson1Composite = compositeImage(btPred1)
btPerson2Composite = compositeImage(btPred2)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "KNN classifier"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "lazy learning method. \n stores the pca vectors of all the \n training faces.\n Classifies a new face by taking \n the majority class of the K faces \n in the training set nearest \n to the test face"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=6,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Conan O Brian"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1.5,fileStream=outputVideo)

image_shaped = imutils.resize(knnPerson1Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Morgan Freeman"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

image_shaped = imutils.resize(knnPerson2Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "SVM classifier"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "eager learning method. \n finds the maximum margin \n hyperplane which separates the classes \n storing only only vectors \n which *support* the hyperplane"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=6,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Conan O Brian"
displayGrid = _write_text(displayGrid,text)
_displayFrame(displayGrid, time=1.5,fileStream=outputVideo)

image_shaped = imutils.resize(svmPerson1Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Morgan Freeman"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1.5,fileStream=outputVideo)

image_shaped = imutils.resize(svmPerson2Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Random Forest classifier"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "an ensemble method which \n predicts the modal prediction of \n a set of decision tree learners"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=3,fileStream=outputVideo)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Conan O Brian"
displayGrid = _write_text(displayGrid,text)
_displayFrame(displayGrid, time=1.5,fileStream=outputVideo)

image_shaped = imutils.resize(btPerson1Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True)

displayGrid =  np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype=np.uint8)
text = "Test Faces predicted as \n Morgan Freeman"
displayGrid = _write_text(displayGrid,text)

_displayFrame(displayGrid, time=1.5,fileStream=outputVideo)

image_shaped = imutils.resize(btPerson2Composite, height=SCREEN_HEIGHT)
imgH,imgW = image_shaped.shape
displayGrid = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH),dtype='int')
displayGrid[0:imgH,0:imgW]=image_shaped
_displayFrame(displayGrid, time=3,fileStream=outputVideo, is_gray=True) 

# L - repeat on more difficult ( non- representitive faces )



outputVideo.release()