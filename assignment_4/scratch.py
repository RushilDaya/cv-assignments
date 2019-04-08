# basically we are going to try and implement a knn classifier here
# this needs to be moved out at some point though 

from shared.utilities import loadLabelled 
from shared.similarityMethods import normalizeSet, getMostLeastSimilar, precisionRecallValidation
import matplotlib.pyplot as plt
import copy
import numpy as np
from shared.utilities import printProgressBar

dataTrain, labelsTrain = loadLabelled('./data/featuresHistogram.pickle')
dataTest,  labelsTest  = loadLabelled('./data/featuresHistogramTest.pickle')
dataTest = dataTest[0:100]
labelsTest = labelsTest[0:100]

normedTraining, normFactors = normalizeSet(dataTrain)
normedTest, _ =  normalizeSet(dataTest, existing_norm_factors=normFactors)

p = precisionRecallValidation(normedTraining, labelsTrain, normedTest, labelsTest, qty_returned=5)