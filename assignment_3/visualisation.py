# implement a number of visualizations of the model
# requires both that a model.pickle and training database exists
import pickle
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import random

def _plot(image):
    # cv2.imshow('0',image)
    # cv2.waitKey(2000)
    plt.imshow(image)
    plt.show()

def buildUpFace(eigenFaces, meanFace, target, max_layers=5):
    # incrementally composes a face from its eigen contributions
    (h,w) = target.shape
    displayGrid = np.zeros((h,w*2))
    displayGrid[:,0:w]=target

    # convert eigenFaces to matrix
    matrix = np.zeros((h*w,len(eigenFaces)))
    for i,face in enumerate(eigenFaces):
        unrolled = np.reshape(face,(h*w,1))
        matrix[:,i] = np.squeeze(unrolled)
    matrix = np.transpose(matrix)
    
    targetMeaned =  target - meanFace
    target_vect = np.reshape(targetMeaned, (h*w,1))
    coeffs = np.matmul(matrix, target_vect)
    _plot(coeffs)

    baseFace = meanFace[:]
    # TODO: add always the most signficant layers
    # not just at random
    max_index = min(max_layers, len(eigenFaces))
    for layer in range(max_index):
        baseFace += coeffs[layer]*eigenFaces[layer]
        displayGrid[:,w:2*w]=baseFace
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