# basically we are going to try and implement a knn classifier here
# this needs to be moved out at some point though 

from shared.utilities import loadLabelled 
import matplotlib.pyplot as plt
import copy
import numpy as np
data, labels = loadLabelled('./data/featuresHistogram.pickle')



means = np.sum(data, axis=0)/data.shape[0]

for colIdx in range(data.shape[1]):
    dimensionMax = np.max(data[:,colIdx])
    data[:,colIdx] = (1.0/dimensionMax)*data[:,colIdx]

testDatum = data[11000,:]
testLabel = labels[11000]

print(testDatum)
print(testLabel)

def getDistance(vectorA, vectorB):
    diff = vectorA - vectorB
    squared = np.square(diff)
    total = np.sum(squared)
    norm = np.sqrt(total)
    return norm

distances = []
for idx in range(data.shape[0]):
    distance= getDistance(testDatum, data[idx,:])
    distances +=[distance]

sortIndices = np.argsort(distances)

sortedLabels = list(np.array(labels)[sortIndices])
plt.plot(sortedLabels[0:200])
plt.show()
