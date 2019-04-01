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

def numMeaningfulVectors(eValues, absolute_max=22):
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
   plt.clf()
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
    
def computeError(TrueImage, PCAs, eigenFaces, meanFace,j):
    
    newFace = copy.deepcopy(meanFace)
    for i in range(len(PCAs)):
        newFace += PCAs[i]*eigenFaces[i]
 

    diff = TrueImage - newFace 
    diff = np.abs(diff)
    total = diff.sum()

    return total


def plotReconstructionError():
    plt.clf()
    data_file = open('./models/eigenfaces.pickle','rb')
    data = pickle.load(data_file)

    eigenFaces = data['eigen_faces']
    meanFace = data['mean_face']


    trainingImages = getTrainingImages()
    meanShifted = [(item - meanFace) for item in trainingImages]

    x,y = meanShifted[0].shape
    length = x*y
    vectors = [np.reshape(item,(length,1)) for item in meanShifted]

    numFaces = len(eigenFaces)
    x,y = eigenFaces[0].shape
    length = x*y

    eigenFaceMatrix = np.zeros((length,numFaces))
    for i in range(numFaces):
        faceColumn = np.reshape(eigenFaces[i], (length,1))
        eigenFaceMatrix[:,i] = np.squeeze(faceColumn)

    eigenFaceMatrix = np.transpose(eigenFaceMatrix) 
    trainingFacePCAs = [np.matmul(eigenFaceMatrix, item) for item in vectors]


    reconError = []
    numCompsVector = []

    for numComps in range(numFaces):
        total_reconstruction_error = 0
        for j in range(len(trainingFacePCAs)):
            facePCA = trainingFacePCAs[j]
            trueImage = trainingImages[j]
            usefulComponents = facePCA[0:numComps]
            usefulEigenFaces = eigenFaces[0:numComps]
            total_reconstruction_error +=computeError(trueImage, usefulComponents, usefulEigenFaces, meanFace,j)

        numCompsVector +=[numComps]
        reconError +=[total_reconstruction_error]

    plt.clf()
    plt.plot(numCompsVector, reconError, color='r')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error on Training Set')
    plt.title('Reconstruction Error')
    plt.savefig('./temp/reconError.png')

    figure = cv2.imread('./temp/reconError.png')
    return figure


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

def visualizeClustering(dimA=0, dimB=1):
    pathTrain = './data/training_faces/'

    plt.clf()
    data_file = open('./models/eigenfaces.pickle','rb')
    data = pickle.load(data_file)

    eigenFaces = data['eigen_faces']
    meanFace = data['mean_face']

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
    groups =('Conan O Brian', 'Morgan Freeman')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)

    plt.savefig('./temp/clusteringTraining'+str(dimA)+str(dimB)+'.png')
    figure = cv2.imread('./temp/clusteringTraining'+str(dimA)+str(dimB)+'.png')

    return figure

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

def _rescale(image):
    return  (255/(np.max(image)-np.min(image)))*(image-np.min(image)).astype('uint8')

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
    max_index = min(max_layers, len(eigenFaces))
    newImage = copy.deepcopy(baseFace)
    images = [newImage]
    for layer in range(max_index):
        baseFace += coeffs[coeffsSortIndices[layer]]*eigenFaces[coeffsSortIndices[layer]]
        plotable = _rescale(baseFace)
        displayGrid[:,w:2*w]=plotable
        newImage = copy.deepcopy(displayGrid)
        images +=[newImage]

    return images

def drawFaceIncrementally():
    trainingFace = loadRandomTrainingFace('./data/training_faces/')

    data_file = open('./models/eigenfaces.pickle','rb')
    data = pickle.load(data_file)
    eigenFaces = data['eigen_faces']
    meanFace = data['mean_face']
    imageList = buildUpFace(eigenFaces, meanFace,trainingFace)

    return imageList
