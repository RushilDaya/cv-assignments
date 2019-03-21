import os
import uuid
from mixins import extractFace
from mixins import getAllImagePaths, getImages, computeEigenFaces, saveModel

PATH_TO_RAW = './data/raw_faces/'
PATH_TO_PROCESSED = './data/extracted_faces/'
PERSONS = ['person_1','person_2']
DIMENSION = 80 # the default


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
        
    allImagePaths = getAllImagePaths(PATH_TO_PROCESSED, recursive=True)
    allFaces = getImages(allImagePaths)
    print('computing eigenfaces ...')
    model = computeEigenFaces(allFaces)
    saveModel(model)