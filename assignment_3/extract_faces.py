# run this script to take all faces in the raw folder
# and generate cropped face images of size N

import numpy as np 
import cv2
import os
import shutil
import uuid

PATH_TO_RAW = './data/raw_faces/'
PATH_TO_PROCESSED = './data/extracted_faces/'
PERSONS = ['person_1','person_2']
DIMENSION = 50 # the default


def _fetchGrayScale(path):
    image = cv2.imread(path)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imageGray

def _detectFace(image):
    # only want to detect a single face per image
    face_cascade = cv2.CascadeClassifier('haar_face_cascade.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) !=1:
        return None
    else:
        return faces[0]

def _cropFace(image, faceCords):
    (x,y,w,h) = faceCords
    face = image[y:y+h,x:x+w]
    return face

def _resizeImage(image, dimension):
    resized = cv2.resize(image, (dimension, dimension), interpolation = cv2.INTER_CUBIC)
    return resized

def _writeToFile(image, pathName):
    imageColor = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(pathName, imageColor)
    return True

def extractFace(size, inputPath, outputPath=None):
    # if outputPath is None the numpy face array
    # is returned by the function
    image = _fetchGrayScale(inputPath)
    faceLocation = _detectFace(image)
    if faceLocation is None:
        print('bad detection on image '+inputPath)
        return False 
    
    cropedFace = _cropFace(image, faceLocation)
    resizedFace = _resizeImage(cropedFace,size)
    if outputPath is None:
        return resizedFace
    else:
        _writeToFile(resizedFace, outputPath)
        return True
    



if __name__ == '__main__':
    for person in PERSONS:
        currentProcessed = os.listdir(PATH_TO_PROCESSED+person)
        for item in currentProcessed:
            os.remove(PATH_TO_PROCESSED+person+'/'+item)
        imageLinks = os.listdir(PATH_TO_RAW+person)
        for image in imageLinks:
            imagePath = PATH_TO_RAW+person+'/'+image
            writePath = PATH_TO_PROCESSED+person+'/'+str(uuid.uuid4())+'.png'
            extractFace(DIMENSION,imagePath, writePath)
        