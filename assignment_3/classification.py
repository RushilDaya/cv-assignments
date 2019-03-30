# this script trains 3 different classifiers
# stores the model weights in pickle files
PATH_TO_FACES = './models/eigenfaces.pickle'
PATH_TO_TRAINING_PERSON_1 = './data/training_faces/person_1/'
PATH_TO_TRAINING_PERSON_2 = './data/training_faces/person_2/'
PATH_TO_KNN_MODEL = './models/knn_model.pickle'
PATH_TO_SVM_MODEL = './models/svm_model.pickle'
PATH_TO_RANDOM_FOREST_MODEL = './models/random_forest_model.pickle'
PATH_TO_TEST_FACES_PERSON_1 = './data/test_faces/person_1/'
PATH_TO_TEST_FACES_PERSON_2 = './data/test_faces/person_2/'
KNN_K = 5

import pickle
import os
import cv2
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# perform KNN classifier

def _importFace(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def _subtractMean(faces1, faces2, meanFace):
    face1New = copy.deepcopy(faces1)
    face2New = copy.deepcopy(faces2)

    face1New = [item-meanFace for item in face1New]
    face2New = [item-meanFace for item in face2New]

    return face1New, face2New

def knnTrain():
    # formats and stores the testing eigenfaces as simple vectors
    pickle_in = open(PATH_TO_FACES,'rb')
    data= pickle.load(pickle_in)
    meanFace = data["mean_face"]
    eigenFaces = data["eigen_faces"]

    facePaths1 = [(PATH_TO_TRAINING_PERSON_1+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_1)]
    facePaths2 = [(PATH_TO_TRAINING_PERSON_2+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_2)]

    faces1 = [_importFace(item) for item in facePaths1 ]
    faces2 = [_importFace(item) for item in facePaths2 ]

    faces1Norm, faces2Norm = _subtractMean(faces1, faces2, meanFace)
    
    #convert the faces into vectors
    x,y = faces1Norm[0].shape
    length = x*y
    reshapedFaces1 = [np.reshape(item, (length,1)) for item in faces1Norm]
    reshapedFaces2 = [np.reshape(item, (length,1)) for item in faces2Norm]

    numEigenFaces = len(eigenFaces)
    x,y = eigenFaces[0].shape
    length = x*y
    eigenFaceMatrix = np.zeros((length,numEigenFaces))
    for i in range(numEigenFaces):
        faceColumn = np.reshape(eigenFaces[i], (length,1))
        eigenFaceMatrix[:,i] = np.squeeze(faceColumn)

    eigenFaceMatrix = np.transpose(eigenFaceMatrix) 
    
    componentFaces1 = [np.matmul(eigenFaceMatrix, item) for item in reshapedFaces1]
    componentFaces2 = [np.matmul(eigenFaceMatrix, item) for item in reshapedFaces2]

    mergedItems = []
    for item in componentFaces1:
        mergedItems+= [(item,1)]

    for item in componentFaces2:
        mergedItems+= [(item,2)]

    model = {}
    model['items'] = mergedItems
    model['mean_face'] = meanFace
    model['pca_components'] = eigenFaceMatrix

    with open(PATH_TO_KNN_MODEL, 'wb') as handle:
        pickle.dump(model,handle)

    return True

def _computeDistance(vect1, vect2):
    if len(vect1) != len(vect2):
        raise TypeError('length mismatch')

    dist = 0.0
    for i in range(len(vect1)):
        dist += (vect1[i][0] - vect2[i][0])**2
        
    dist = math.sqrt(dist)
    
    return dist
    

def knnClassify(imagePath):

    pickle_in = open(PATH_TO_KNN_MODEL,'rb')
    data= pickle.load(pickle_in)

    trainItems = data['items']
    meanFace = data['mean_face']
    pcaComponents = data['pca_components']

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageNorm = image - meanFace
    x,y = imageNorm.shape
    length = x*y

    vectorized = np.reshape(imageNorm,(length,1))

    pcaComps = np.matmul(pcaComponents, vectorized)

    distances = []
    for item in trainItems:
        weights, label = item 
        distance = _computeDistance(weights, pcaComps)
        distances +=[(distance ,label)]

    distances.sort(key=lambda tup: tup[0], reverse=False)
    
    distances = distances[0:KNN_K]

    vote_1 = 0
    vote_2 = 0

    for item in distances:
        _,label = item 
        if label == 1:
            vote_1 +=1
        else:
            vote_2 +=1

    if vote_1 > vote_2:
        return 1 
    else:
        return 2

def knnEvaluate():
    # get the accuracy of the knn classifier
    person1 = [PATH_TO_TEST_FACES_PERSON_1+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_1)]
    person2 = [PATH_TO_TEST_FACES_PERSON_2+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_2)]

    predictions1 = [knnClassify(item) for item in person1] 
    predictions2 = [knnClassify(item) for item in person2]

    correct = 0
    for item in predictions1:
        if item == 1:
            correct+=1
    for item in predictions2:
        if item == 2:
            correct+=1
    
    accuracy = float(correct)/(len(predictions1)+len(predictions2))

    return accuracy

def svmTrain():
    # first we need to run pca on the test data
    # obtain an array of pca vectors 
    pickle_in = open(PATH_TO_FACES,'rb')
    data= pickle.load(pickle_in)
    meanFace = data["mean_face"]
    eigenFaces = data["eigen_faces"]

    facePaths1 = [(PATH_TO_TRAINING_PERSON_1+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_1)]
    facePaths2 = [(PATH_TO_TRAINING_PERSON_2+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_2)]

    faces1 = [_importFace(item) for item in facePaths1 ]
    faces2 = [_importFace(item) for item in facePaths2 ]

    faces1Norm, faces2Norm = _subtractMean(faces1, faces2, meanFace)
    
    #convert the faces into vectors
    x,y = faces1Norm[0].shape
    length = x*y
    reshapedFaces1 = [np.reshape(item, (length,1)) for item in faces1Norm]
    reshapedFaces2 = [np.reshape(item, (length,1)) for item in faces2Norm]

    numEigenFaces = len(eigenFaces)
    x,y = eigenFaces[0].shape
    length = x*y
    eigenFaceMatrix = np.zeros((length,numEigenFaces))
    for i in range(numEigenFaces):
        faceColumn = np.reshape(eigenFaces[i], (length,1))
        eigenFaceMatrix[:,i] = np.squeeze(faceColumn)

    eigenFaceMatrix = np.transpose(eigenFaceMatrix) 
    
    componentFaces1 = [np.transpose(np.matmul(eigenFaceMatrix, item)).tolist()[0] for item in reshapedFaces1]
    componentFaces2 = [np.transpose(np.matmul(eigenFaceMatrix, item)).tolist()[0] for item in reshapedFaces2]

    mergedItems = []
    labels = []
    for item in componentFaces1:
        mergedItems+= [item]
        labels+=[1]
    for item in componentFaces2:
        mergedItems+= [item]
        labels+=[2]

    # perform the training
    classifier = svm.SVC(gamma='scale')
    classifier.fit(mergedItems,labels)

    # save the classifier
    model = {}
    model['mean_face'] = meanFace
    model['pca_components'] = eigenFaceMatrix
    model['classifier']=classifier

    with open(PATH_TO_SVM_MODEL, 'wb') as handle:
        pickle.dump(model,handle)

    return True

def svmPredict(imgPath):
    # load the classifier
    pickle_in = open(PATH_TO_SVM_MODEL,'rb')
    data= pickle.load(pickle_in)

    svmClassifier = data['classifier']
    meanFace = data['mean_face'] 
    pcaComponents = data['pca_components']

    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageNorm = image - meanFace
    x,y = imageNorm.shape
    length = x*y

    vectorized = np.reshape(imageNorm,(length,1))

    pcaComps = np.transpose(np.matmul(pcaComponents, vectorized)).tolist()
    # pipeline the test item
    prediction = svmClassifier.predict(pcaComps)
    # prediction
    return  prediction[0]

def svmEvaluate():
    person1 = [PATH_TO_TEST_FACES_PERSON_1+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_1)]
    person2 = [PATH_TO_TEST_FACES_PERSON_2+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_2)]

    predictions1 = [svmPredict(item) for item in person1] 
    predictions2 = [svmPredict(item) for item in person2]

    correct = 0
    for item in predictions1:
        if item == 1:
            correct+=1
    for item in predictions2:
        if item == 2:
            correct+=1
    
    accuracy = float(correct)/(len(predictions1)+len(predictions2))

    return accuracy

def boostedTreeTrain():
 # first we need to run pca on the test data
    # obtain an array of pca vectors 
    pickle_in = open(PATH_TO_FACES,'rb')
    data= pickle.load(pickle_in)
    meanFace = data["mean_face"]
    eigenFaces = data["eigen_faces"]

    facePaths1 = [(PATH_TO_TRAINING_PERSON_1+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_1)]
    facePaths2 = [(PATH_TO_TRAINING_PERSON_2+item) for item in os.listdir(PATH_TO_TRAINING_PERSON_2)]

    faces1 = [_importFace(item) for item in facePaths1 ]
    faces2 = [_importFace(item) for item in facePaths2 ]

    faces1Norm, faces2Norm = _subtractMean(faces1, faces2, meanFace)
    
    #convert the faces into vectors
    x,y = faces1Norm[0].shape
    length = x*y
    reshapedFaces1 = [np.reshape(item, (length,1)) for item in faces1Norm]
    reshapedFaces2 = [np.reshape(item, (length,1)) for item in faces2Norm]

    numEigenFaces = len(eigenFaces)
    x,y = eigenFaces[0].shape
    length = x*y
    eigenFaceMatrix = np.zeros((length,numEigenFaces))
    for i in range(numEigenFaces):
        faceColumn = np.reshape(eigenFaces[i], (length,1))
        eigenFaceMatrix[:,i] = np.squeeze(faceColumn)

    eigenFaceMatrix = np.transpose(eigenFaceMatrix) 
    
    componentFaces1 = [np.transpose(np.matmul(eigenFaceMatrix, item)).tolist()[0] for item in reshapedFaces1]
    componentFaces2 = [np.transpose(np.matmul(eigenFaceMatrix, item)).tolist()[0] for item in reshapedFaces2]

    mergedItems = []
    labels = []
    for item in componentFaces1:
        mergedItems+= [item]
        labels+=[1]
    for item in componentFaces2:
        mergedItems+= [item]
        labels+=[2]

    # perform the training
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(mergedItems,labels)

    # save the classifier
    model = {}
    model['mean_face'] = meanFace
    model['pca_components'] = eigenFaceMatrix
    model['classifier']=classifier

    with open(PATH_TO_RANDOM_FOREST_MODEL, 'wb') as handle:
        pickle.dump(model,handle)

    return True

def boostedTreePredict(imgPath):
    # load the classifier
    pickle_in = open(PATH_TO_RANDOM_FOREST_MODEL,'rb')
    data= pickle.load(pickle_in)

    rfClassifier = data['classifier']
    meanFace = data['mean_face'] 
    pcaComponents = data['pca_components']

    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageNorm = image - meanFace
    x,y = imageNorm.shape
    length = x*y

    vectorized = np.reshape(imageNorm,(length,1))

    pcaComps = np.transpose(np.matmul(pcaComponents, vectorized)).tolist()
    # pipeline the test item
    prediction = rfClassifier.predict(pcaComps)
    # prediction
    return  prediction[0]

def boostedTreeEvaluate():
    person1 = [PATH_TO_TEST_FACES_PERSON_1+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_1)]
    person2 = [PATH_TO_TEST_FACES_PERSON_2+item for item in os.listdir(PATH_TO_TEST_FACES_PERSON_2)]

    predictions1 = [boostedTreePredict(item) for item in person1] 
    predictions2 = [boostedTreePredict(item) for item in person2]

    correct = 0
    for item in predictions1:
        if item == 1:
            correct+=1
    for item in predictions2:
        if item == 2:
            correct+=1
    
    accuracy = float(correct)/(len(predictions1)+len(predictions2))

    return accuracy

if __name__ == '__main__':
    knnTrain()
    acc = knnEvaluate()
    print('knn accuracy: ',str(acc))
    svmTrain()
    acc = svmEvaluate()
    print('svm accuracy: ', str(acc))
    boostedTreeTrain()
    acc = boostedTreeEvaluate()
    print('boosted tree accuracy: ',str(acc))