# determines the eigenfaces of the space
import numpy as np 
from numpy import linalg as LA
import cv2 
import os
import matplotlib.pyplot as plt 
import pickle

PATH_TO_PROCESSED = './data/extracted_faces/'


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

def __unroll(matrix):
    (x,y) = matrix.shape
    values = x*y
    return np.reshape(matrix,(values,1))


def __roll(vector, N):
    return np.reshape(vector,(N,N))

def __numMeaningfulVectors(eValues, absolute_max=20):
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
    vectors = [(__unroll(face)) for face in meanShifedFaces]
    A = np.zeros((H*H,numFaces), dtype='float')
    for i in range(numFaces):
        A[:,i] = np.squeeze(vectors[i])

    covarianceMatrix = np.matmul(A,np.transpose(A))
    eVals, eVects = LA.eig(covarianceMatrix)
    numImportant = __numMeaningfulVectors(eVals)
    
    eigenFaces = []
    for i in range(numImportant):
        face = __roll(eVects[:,i],H)
        eigenFaces.append(face.real)

    model = {}
    model['eigen_faces']=eigenFaces
    model['image_size']=H 
    model['train_size']=numFaces

    return eigenFaces

def saveModel(model):
    with open('model.pickle', 'wb') as handle:
        pickle.dump(model,handle)
    return True

if __name__ =='__main__':
    allImagePaths = getAllImagePaths(PATH_TO_PROCESSED, recursive=True)
    allFaces = getImages(allImagePaths)
    model = computeEigenFaces(allFaces)
    saveModel(model)
