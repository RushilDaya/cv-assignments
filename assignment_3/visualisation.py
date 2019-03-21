# implement a number of visualizations of the model
# requires both that a model.pickle and training database exists
import pickle
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import copy
import random


def _rescale(image):
    return  (1.0/(np.max(image)-np.min(image)))*(image-np.min(image)).astype('uint8')

def _plot(image, p_type='CV2'):
    if p_type == 'CV2':
        h,w = image.shape 
        if w < 1000:
            scale = int(1000.00/w) 
        imageBig = cv2.resize(image,(w*scale,h*scale))
        cv2.imshow('0',imageBig)
        cv2.waitKey(100)
    else:
        plt.imshow(image)
        plt.show()

def calculateError(face1, face2):
    difference = face1-face2
    absError = np.square(difference)
    a = np.sqrt(np.sum(absError))
    return a

def buildUpFace(eigenFaces, meanFace, target, max_layers=100):
    # incrementally composes a face from its eigen contributions
    (h,w) = target.shape
    displayGrid = np.zeros((h,w*2))
    plotableTarget = _rescale(target)
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
    plotable = _rescale(baseFace)
    displayGrid[:,w:2*w]=plotable 
    _plot(displayGrid)
    max_index = min(max_layers, len(eigenFaces))
    for layer in range(max_index):
        baseFace += coeffs[coeffsSortIndices[layer]]*eigenFaces[coeffsSortIndices[layer]]
        print(calculateError(baseFace, target))
        plotable = _rescale(baseFace)
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

def _getImageVector(paths, meanFace):
    images = [(cv2.imread(path)) for path in paths]
    images = [(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for image in images]
    images = [image-meanFace for image in images]
    (w,h) = images[0].shape
    length = w*h

    tempMatrix = np.zeros((length, len(images)))
    for idx,image in enumerate(images):
        imageFlat = np.reshape(image,(length,1))
        tempMatrix[:, idx] = np.squeeze(imageFlat)
    return tempMatrix

def _getCoeffs(targetMatrix, eigenFaces):
    (w,h) = eigenFaces[0].shape
    length = w*h

    EigenVectors = np.zeros((length, len(eigenFaces)))
    for idx, image in enumerate(eigenFaces):
        imageFlat = np.reshape(image, (length,1))
        EigenVectors[:, idx] = np.squeeze(imageFlat)
    EigenVectors = np.transpose(EigenVectors)

    coeffs = np.transpose(np.matmul(EigenVectors, targetMatrix))
    return coeffs

def _extractCol(matrix, col_id):
    temp = np.transpose(matrix[col_id,:])
    temp = temp.tolist()
    return temp

def twoDimPlot(pathTrain, eigenFaces, meanFace, dimA=0, dimB=1):
    # will make a scatter plot in 2d based on the selected features

    # get two vectors of training images 
    # compute from them 2 separate coefficient representations

    person_1_paths = [ pathTrain+'/person_1/'+item for item in os.listdir(pathTrain+'/person_1/')]
    person_2_paths = [ pathTrain+'/person_2/'+item for item in os.listdir(pathTrain+'/person_2/')]

    P1 = _getImageVector(person_1_paths, meanFace)
    P2 = _getImageVector(person_2_paths, meanFace)

    P1_COEFFS = _getCoeffs(P1, eigenFaces)
    P2_COEFFS = _getCoeffs(P2, eigenFaces)

    P1_DATA = (_extractCol(P1_COEFFS, dimA), _extractCol(P1_COEFFS, dimB))
    P2_DATA = (_extractCol(P2_COEFFS, dimA), _extractCol(P2_COEFFS, dimB))
    data = (P1_DATA, P2_DATA)
    colors = ('green', 'red')
    groups =('person 1', 'person 2')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

    return True



if __name__ == '__main__':
    eigenFaces, meanFace = loadModel('./models/eigenfaces.pickle')
    randomTrainingFace = loadRandomTrainingFace('./data/training_faces')
    buildUpFace(copy.deepcopy(eigenFaces), meanFace, randomTrainingFace)
    twoDimPlot('./data/training_faces', copy.deepcopy(eigenFaces), meanFace, 1,5)
