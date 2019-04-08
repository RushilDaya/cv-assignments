import numpy as np 
import math
import matplotlib.pyplot as plt
from shared.utilities import printProgressBar

def _histogramOneColor(array,resolution):
    # performs the actual bucketing process
    bucket_width = 255/resolution
    buckets = np.zeros((1,resolution),dtype='float')

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            index = math.floor(array[i,j]/bucket_width)-1
            buckets[0,index]+=1
    return buckets


def _performHistogramBucketing(image, resolution, numColors):
    featureRow = np.zeros((1,resolution*numColors))
    for colorIdx in range(numColors):
        featureRow[0,colorIdx*resolution:colorIdx*resolution+resolution]=_histogramOneColor(image[:,:,colorIdx],resolution)
    return featureRow

def histogramFeatures(images, labels, bucket_resolution=20 ):
    # construct a global histogram
    

    numImages = images.shape[0]
    colorDepth = images.shape[3]
    featureArray = np.zeros((numImages,colorDepth*bucket_resolution),dtype='float')
    for idx in range(numImages):
        featureArray[idx,:]=_performHistogramBucketing(images[idx,:,:,:], bucket_resolution, colorDepth)
        printProgressBar(idx,numImages,prefix='progress ')
    
    return  featureArray, labels