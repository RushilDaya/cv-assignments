# implement a number of visualizations of the model
# requires both that a model.pickle and training database exists
import pickle
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import random

def _plot(image):
    h,w = image.shape 
    if w < 1000:
        scale = int(1000.00/w) 
    imageBig = cv2.resize(image,(w*scale,h*scale))
    cv2.imshow('0',imageBig)
    cv2.waitKey(500)
    # plt.imshow(image)
    # plt.show()

def calculateError(face1, face2):
    difference = face1-face2
    absError = np.square(difference)
    a = np.sqrt(np.sum(absError))
    return a

def buildUpFace(eigenFaces, meanFace, target, max_layers=100):
    # incrementally composes a face from its eigen contributions
    (h,w) = target.shape
    displayGrid = np.zeros((h,w*2))
    plotableTarget = (1.0/(np.max(target)-np.min(target)))*(target-np.min(target)).astype('uint8')
    displayGrid[:,0:w]=plotableTarget

    # convert eigenFaces to matrix
    matrix = np.zeros((h*w,len(eigenFaces)))
    for i,face in enumerate(eigenFaces):
        unrolled = np.reshape(face,(h*w,1))
        matrix[:,i] = np.squeeze(unrolled)
    matrix = np.transpose(matrix)
    
    targetMeaned =  target - meanFace
    target_vect = np.reshape(targetMeaned, (h*w,1))
    coeffs = np.transpose(np.matmul(matrix, target_vect))[0]

    coeffsSortIndices = np.argsort(-1*abs(coeffs))

    baseFace = meanFace[:]
    plotable = (1.0/(np.max(baseFace)-np.min(baseFace)))*(baseFace-np.min(baseFace)).astype('uint8')
    displayGrid[:,w:2*w]=plotable 
    _plot(displayGrid)
    max_index = min(max_layers, len(eigenFaces))
    for layer in range(max_index):
        baseFace += coeffs[coeffsSortIndices[layer]]*eigenFaces[coeffsSortIndices[layer]]
        print(calculateError(baseFace, target))
        plotable = (1.0/(np.max(baseFace)-np.min(baseFace)))*(baseFace-np.min(baseFace)).astype('uint8')
        displayGrid[:,w:2*w]=plotable
        _plot(displayGrid)

    return True

def loadModel(path):
    pickle_in = open(path,'rb')
    data = pickle.load(pickle_in)
    return data["eigen_faces"], data["mean_face"]

def loadRandomTrainingFace(path):
    persons = os.listdir(path)
    trainImagePaths = []
    for person in persons:
        imagePaths = os.listdir(path+'/'+person)
        imagePaths = [path+'/'+person+'/'+item for item in imagePaths]
        trainImagePaths +=imagePaths
    index = random.randint(0, len(trainImagePaths))
    selection = trainImagePaths[index]
    image = cv2.imread(selection)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
    

if __name__ == '__main__':
    eigenFaces, meanFace = loadModel('model.pickle')
    randomTrainingFace = loadRandomTrainingFace('./data/extracted_faces')
    buildUpFace(eigenFaces, meanFace, randomTrainingFace)
    