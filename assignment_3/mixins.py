import numpy as np 
from numpy import linalg as LA
import cv2
import os
import shutil
import uuid
import matplotlib.pyplot as plt 
import pickle
import random
import copy

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
    model['eigen_values']=eVals

    return model

def saveModel(model):
    with open('models/eigenfaces.pickle', 'wb') as handle:
        pickle.dump(model,handle)
    return True


def splitSets(pathData,pathTraining, pathTest, numTrain, numTest):
    # restructures the data into a training and test set
    persons = os.listdir(pathData)
    for person in persons:
        num_extracted = len(os.listdir(pathData+person))
        if num_extracted < numTest + numTrain:
            raise TypeError('not enough faces in database')
    
    oldImages = []
    for person in persons:
        oldImages = oldImages + [pathTraining+person+'/'+item for item in os.listdir(pathTraining+person)]
        oldImages = oldImages + [pathTest+person+'/'+item for item in os.listdir(pathTest+person)]
    [(os.remove(item)) for item in oldImages]

    personOneNew = os.listdir(pathData+'person_1')
    personTwoNew = os.listdir(pathData+'person_2')

    personOneTrain = personOneNew[0:numTrain]
    personOneTest = personOneNew[numTrain:numTrain+numTest]
    personTwoTrain = personTwoNew[0:numTrain]
    personTwoTest = personTwoNew[numTrain:numTrain+numTest]

    for item in personOneTrain:
        shutil.move(pathData+'person_1/'+item, pathTraining+'person_1/'+item)
    
    for item in personOneTest:
        shutil.move(pathData+'person_1/'+item, pathTest+'person_1/'+item)

    for item in personTwoTrain:
        shutil.move(pathData+'person_2/'+item, pathTraining+'person_2/'+item)

    for item in personTwoTest:
        shutil.move(pathData+'person_2/'+item, pathTest+'person_2/'+item)


def getRawImages(num_images=1, sample_type='RANDOM',imFormat='GRAYSCALE'):
    possible_directories = ['./data/raw_faces/person_1/','./data/raw_faces/person_2/']
    returnedArray = []
    for item in range(num_images):
        if sample_type=='RANDOM':
            path = random.choice(possible_directories)
            images = os.listdir(path)
            randomImage = random.choice(images)
            imagePath = path+randomImage
        else:
            raise TypeError('method not implemented')
        
        image = cv2.imread(imagePath)
        
        if imFormat=='GRAYSCALE':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        returnedArray +=[image]
    
    return returnedArray

def getTrainingImages():
    directories = ['./data/training_faces/person_1/','./data/training_faces/person_2/']
    returnedArray = []
    for path in directories:
        imageNames =os.listdir(path)
        fullPaths = [path+item for item in imageNames]
        images = [(cv2.imread(item)) for item in fullPaths]
        returnedArray +=images 
    returnedArray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image  in returnedArray]
    return returnedArray

def getMeanFace():
    data_file = open('./models/eigenfaces.pickle','rb')
    data = pickle.load(data_file)
    return data['mean_face']

def getEigenFaces(num_items=1, formated = True):
    data_file = open('./models/eigenfaces.pickle','rb')
    data = pickle.load(data_file)
    eigenFaces = data['eigen_faces'][0:num_items]

    trueMax = max([np.max(item) for item in eigenFaces])
    trueMin = min([np.min(item) for item in eigenFaces])
    print(trueMax)
    print(trueMin)

    returnedArray = []
    for face in eigenFaces:
        height,width = face.shape
        positives = copy.deepcopy(face)
        negatives = copy.deepcopy(face)

        positives[positives<0] = 0
        negatives[negatives>0] = 0

        positives = (255/trueMax)*positives
        negatives = (200/trueMin)*negatives

        blank_image = np.zeros((height,width,3), np.uint8)
        blank_image[:,:,1]=positives
        blank_image[:,:,2]=negatives
        returnedArray +=[blank_image]
    return returnedArray

def plotEigenValues():
   data_file = open('./models/eigenfaces.pickle','rb')
   data = pickle.load(data_file)
   evals = data['eigen_values']
   evals = [abs(item) for item in evals]

   linearScale = [item for item in range(len(evals))]

   plt.plot(linearScale, evals, color='r')
   plt.xlabel('PCA eigenvalue number')
   plt.ylabel('PCA eigenvalue (absolute size)')
   plt.title('Plot of eigenvalue size from PCA')
   plt.savefig('./temp/eigenvalues.png')

   figure = cv2.imread('./temp/eigenvalues.png')
   return figure
    
