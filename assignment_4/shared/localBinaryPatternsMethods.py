import numpy as np 
import math
import matplotlib.pyplot as plt
from shared.configurationParser import retrieveConfiguration as rCon
from skimage.feature import local_binary_pattern
from shared.utilities import printProgressBar


def _flatten(data):
    colorDepth = data.shape[3]
    flattened = np.zeros((data.shape[0],data.shape[1],data.shape[2]),dtype='float')
    for idx in range(colorDepth):
        flattened= flattened + data[:,:,:,idx]/3
    return flattened

def _histogram(image, histogram_resolution):
    buckets = np.zeros((256), dtype='float')
    dimA,dimB = image.shape
    for i in range(dimA):
        for j in range(dimB):
            buckets[int(image[i,j])]+=1

    bucketResized = np.resize(buckets,(histogram_resolution))
    return bucketResized


def _lbfCompute(image, histogram_resolution, points, radius):
    lbp = local_binary_pattern(image, points, radius)
    histo = _histogram(lbp, histogram_resolution)
    return histo


def lbpFeatures(images, labels):
    lbp_bucket_resolution = rCon('LBP_BUCKET_RESOLUTION')
    lbp_points = rCon('LBP_POINTS')
    lbp_radius = rCon('LBP_RADIUS')

    numImages = images.shape[0]
    colorDepth = images.shape[3]
    flat_images = _flatten(images)
    globalHistogram = np.zeros((numImages, lbp_bucket_resolution), dtype='float')
    for idx in range(numImages):
        printProgressBar(idx, numImages)
        globalHistogram[idx,:] = _lbfCompute(flat_images[idx,:,:], lbp_bucket_resolution, lbp_points, lbp_radius)

    return globalHistogram, labels
