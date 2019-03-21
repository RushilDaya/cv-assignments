import os
import uuid
from mixins import extractFace
from mixins import getAllImagePaths, getImages, computeEigenFaces, saveModel, splitSets

PATH_TO_RAW = './data/raw_faces/'
PATH_TO_PROCESSED = './data/extracted_faces/'
PATH_TO_TRAINING = './data/training_faces/'
PATH_TO_TEST = './data/test_faces/'
PERSONS = ['person_1','person_2']
DIMENSION = 50 # the default
TRAIN_NUM = 10
TEST_NUM = 5

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

    splitSets(PATH_TO_PROCESSED,PATH_TO_TRAINING,PATH_TO_TEST, TRAIN_NUM, TEST_NUM )
    allImagePaths = getAllImagePaths(PATH_TO_TRAINING, recursive=True)
    allFaces = getImages(allImagePaths)
    print('computing eigenfaces ...')
    model = computeEigenFaces(allFaces)
    saveModel(model)