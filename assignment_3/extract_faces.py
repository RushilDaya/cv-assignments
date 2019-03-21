# run this script to take all faces in the raw folder
# and generate cropped face images of size N

import numpy as np 
from numpy import linalg as LA
import cv2
import os
import shutil
import uuid
import matplotlib.pyplot as plt 
import pickle
from mixins import extractFace

PATH_TO_RAW = './data/raw_faces/'
PATH_TO_PROCESSED = './data/extracted_faces/'
PERSONS = ['person_1','person_2']
DIMENSION = 50 # the default


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
        
