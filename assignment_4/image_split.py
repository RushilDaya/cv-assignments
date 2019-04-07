'''
    loads data from the cifar dataset and splits it into
    a training set, a test set and a play set
'''

import numpy as np 
from keras.datasets import cifar10
from shared.configurationParser import retrieveConfiguration as rConfig
from shared.errors import ConfigurationFileError, ProbabilisticError
from shared.distributionChecks import uniformDistribution
from shared.utilities import saveLabelled


(xA,yA),(xB,yB) = cifar10.load_data()

xData = np.concatenate((xA,xB))
yData = np.concatenate((yA,yB))

dataSize = xData.shape[0]

numTraining = int( dataSize*rConfig('DATA_SPLIT_TRAINING') )
numTest = int( dataSize*rConfig('DATA_SPLIT_TEST') )
numPlay = int( dataSize*rConfig('DATA_SPLIT_PLAY') )

if numPlay + numTest + numTraining != dataSize:
    raise ConfigurationFileError()

Idx = 0
trainData = xData[Idx:Idx+numTraining,:,:,:]
trainLabels = yData[Idx:Idx+numTraining,:]
Idx +=numTraining
testData = xData[Idx:Idx+numTest,:,:,:]
testLabels = yData[Idx:Idx+numTest,:]
Idx +=numTest
playData = xData[Idx:Idx+numPlay,:,:,:]
playLabels = yData[Idx:Idx+numPlay,:]

# check to see if all the sets have uniform class distributions
nc = rConfig('DATA_NUM_CLASSES')

if not uniformDistribution(np.transpose(trainLabels)[0],nc) or not uniformDistribution(np.transpose(testLabels)[0],nc) or not uniformDistribution(np.transpose(playLabels)[0],nc):
    raise ProbabilisticError('unrepresentative data partition: try rerun or change threshold')


trainDataPath = rConfig('DATA_NAME_TRAINING')
testDataPath =  rConfig('DATA_NAME_TEST')
playDataPath =  rConfig('DATA_NAME_PLAY')

saveLabelled(trainDataPath, trainData, trainLabels)
saveLabelled(testDataPath, testData, testLabels)
saveLabelled(playDataPath, playData, playLabels)
