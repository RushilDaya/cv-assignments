import numpy as np 
from numpy import linalg as LA
import cv2
import os
import shutil
import uuid
import matplotlib.pyplot as plt 
import pickle

def readGrayscale(path):
    image = cv2.imread(path)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imageGray

def detectFace(image):
    # only want to detect a single face per image
    face_cascade = cv2.CascadeClassifier('models/haar_face_cascade.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) !=1:
        return None
    else:
        return faces[0]

def cropFace(image, faceCords):
    (x,y,w,h) = faceCords
    face = image[y:y+h,x:x+w]
    return face

def resizeImageSquare(image, dimension):
    resized = cv2.resize(image, (dimension, dimension), interpolation = cv2.INTER_CUBIC)
    return resized

def writeImage(image, pathName):
    imageColor = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(pathName, imageColor)
    return True

def extractFace(size, inputPath, outputPath=None):
    # if outputPath is None the numpy face array
    # is returned by the function
    image = readGrayscale(inputPath)
    faceLocation = detectFace(image)
    if faceLocation is None:
        print('bad detection on image '+inputPath)
        return False 
    
    cropedFace = cropFace(image, faceLocation)
    resizedFace = resizeImageSquare(cropedFace,size)
    if outputPath is None:
        return resizedFace
    else:
        writeImage(resizedFace, outputPath)
        return True

def getAllImagePaths(path, recursive=False,append_path=True):
    allPaths = os.listdir(path)
    imagePaths = [item for item in allPaths if os.path.isdir(path+'/'+item) != True]
    folders = [item for item in allPaths if os.path.isdir(path+'/'+item) == True]
    
    if recursive == True:
        for folder in folders:
            tempImagePaths = getAllImagePaths(path+'/'+folder, recursive=True, append_path=False)
            tempImagePaths = [folder+'/'+imagePath for imagePath in tempImagePaths]
            imagePaths +=tempImagePaths

    if append_path == True:
        imagePaths = [path+item for item in imagePaths]
    return imagePaths

def getImages(pathList):
    images = [(cv2.imread(item)) for item in pathList]
    images = [(cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)) for item in images]
    return images

def unroll(matrix):
    (x,y) = matrix.shape
    values = x*y
    return np.reshape(matrix,(values,1))


def roll(vector, N):
    return np.reshape(vector,(N,N))

def numMeaningfulVectors(eValues, absolute_max=20):
    # the smaller the eval the less important the corr. evec
    # here we try to limit the number of meaningful eVecs
    
    # TODO: implement a knee finder

    return absolute_max

def computeEigenFaces(faceList):
    # get the mean face from the list
    numFaces = len(faceList)
    (H,W) = faceList[0].shape
    if H != W:
        raise TypeError('Dimension Mismatch')

    accumulator = np.zeros( (H,W), dtype='float')
    for face in faceList:
        accumulator +=face 
    meanFace = accumulator/numFaces

    meanShifedFaces = [face - meanFace for face in faceList]
    vectors = [(unroll(face)) for face in meanShifedFaces]
    A = np.zeros((H*H,numFaces), dtype='float')
    for i in range(numFaces):
        A[:,i] = np.squeeze(vectors[i])

    covarianceMatrix = np.matmul(A,np.transpose(A))
    eVals, eVects = LA.eig(covarianceMatrix)
    numImportant = numMeaningfulVectors(eVals)
    
    eigenFaces = []
    for i in range(numImportant):
        face = roll(eVects[:,i],H)
        eigenFaces.append(face.real)

    model = {}
    model['mean_face'] = meanFace
    model['eigen_faces']=eigenFaces
    model['image_size']=H 
    model['train_size']=numFaces

    return model

def saveModel(model):
    with open('models/eigenfaces.pickle', 'wb') as handle:
        pickle.dump(model,handle)
    return True