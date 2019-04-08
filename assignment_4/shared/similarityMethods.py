''' functions dealing with computing similarity between items
provides enough information back for the purposes of reporting
'''

import copy
import numpy as np 


def normalizeSet(vectorRows, existing_norm_factors=None):
    # assumes each row is a sample
    normalizedRows = copy.deepcopy(vectorRows)
    use_existing_norms = (existing_norm_factors !=None)

    if use_existing_norms:
        normFactors = existing_norm_factors
    else:
        normFactors = []
        for colIdx in range(normalizeVectors.shape[1]):
            _max = np.max(normalizeRows[:,colIdx])
            normFactor = 1.0/_max
            normFactors += [normFactor]

    for colIdx in range(normalizeVectors.shape[1]):
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
    leastSim = sortedIndices[-qty_least:-1]
    mostSim_values = list(np.array(similarities)[mostSim])
    leastSim_values = list(np.array(similarities)[leastSim])

    return mostSim, leastSim, mostSim_values, leastSim_values
