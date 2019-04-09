''' functions dealing with computing similarity between items
provides enough information back for the purposes of reporting
'''

import copy
import numpy as np 
from shared.utilities import printProgressBar


def normalizeSet(vectorRows, existing_norm_factors=None):
    # assumes each row is a sample
    normalizedRows = copy.deepcopy(vectorRows)
    use_existing_norms = (existing_norm_factors !=None)

    if use_existing_norms:
        normFactors = existing_norm_factors
    else:
        normFactors = []
        for colIdx in range(normalizedRows.shape[1]):
            _max = np.max(normalizedRows[:,colIdx])
            normFactor = 1.0/_max
            normFactors += [normFactor]

    for colIdx in range(normalizedRows.shape[1]):
        normalizedRows[:,colIdx] = normFactors[colIdx]*normalizedRows[:, colIdx]
    
    return normalizedRows,normFactors

def vectorSimilarity(v1, v2):
    diff = v1 - v2
    sqrVect = np.square(diff)
    total = np.sum(sqrVect)
    similarity = np.sqrt(total)
    return similarity

def getMostLeastSimilar(dbRows, inputVect, qty_most=10 , qty_least=10  ):
    ''' returns indices which can be used to look up in the db array'''
    similarities = []
    for idx in range(dbRows.shape[0]):
        similarity = vectorSimilarity(inputVect, dbRows[idx,:])
        similarities +=[similarity]

    sortedIndices = np.argsort(similarities)

    mostSim = sortedIndices[0:qty_most]
    leastSim = sortedIndices[-qty_least-1:-1]
    mostSim_values = list(np.array(similarities)[mostSim])
    leastSim_values = list(np.array(similarities)[leastSim])

    return mostSim, leastSim, mostSim_values, leastSim_values

def precisionRecallValidation(trainingData,trainingLabels, validationData, validationLabels, qty_returned=50, num_classes=10):
    ''' returns a mean precision and recall'''
    ''' assume uniform distribution of classes'''

    same_class_returned_tally = 0
    validation_set_size = validationData.shape[0]

    for idx in range(validation_set_size):
        printProgressBar(idx,validation_set_size)
        mSimTrain,_,_,_=getMostLeastSimilar(trainingData,validationData[idx,:],qty_most=qty_returned)
        mSimLabels=np.transpose(trainingLabels[mSimTrain]).tolist()[0]
        numCorrect=mSimLabels.count(validationLabels[idx])
        same_class_returned_tally +=numCorrect

    meanPrecision = float(same_class_returned_tally)/(qty_returned*validation_set_size)
    meanRecall = float(same_class_returned_tally*num_classes)/(validation_set_size*trainingData.shape[0])
    return meanPrecision, meanRecall
