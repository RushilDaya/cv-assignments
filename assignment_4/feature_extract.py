import numpy as np 
from shared.configurationParser import retrieveConfiguration as rCon 
from shared.utilities import loadLabelled, saveLabelled,saveObj
from shared.histogramFeatureMethods import histogramFeatures
from shared.convnetFeatureMethods import learnConvNet, convNetFeatures

trainingDataPath = rCon('DATA_NAME_TRAINING')
testDataPath = rCon('DATA_NAME_TEST')


if rCon('RUN_HISTOGRAM_FEATURE_EXTRACT'):
    loadedFeatures, loadedLabels = loadLabelled(trainingDataPath)
    extractedFeatures,extractedLabels = histogramFeatures(loadedFeatures,loadedLabels,bucket_resolution=rCon('HISTOGRAM_BUCKET_RESOLUTION'))
    saveLabelled(rCon('HISTOGRAM_FEATURE_PATH_TRAINING'),extractedFeatures,extractedLabels)

    loadedFeatures, loadedLabels = loadLabelled(testDataPath)
    extractedFeatures,extractedLabels = histogramFeatures(loadedFeatures,loadedLabels,bucket_resolution=rCon('HISTOGRAM_BUCKET_RESOLUTION'))
    saveLabelled(rCon('HISTOGRAM_FEATURE_PATH_TEST'),extractedFeatures,extractedLabels)

if rCon('RUN_CONVNET_FEATURE_EXTRACT'):
    loadedFeatures, loadedLabels = loadLabelled(trainingDataPath)
    model = learnConvNet(loadedFeatures,loadedLabels,rCon('DATA_NUM_CLASSES'))
    saveObj(rCon('CONVNET_MODEL_PATH'),model)

    extractedFeatures,extractedLabels = convNetFeatures(loadedFeatures,loadedLabels,model=model)
    saveLabelled(rCon('CONVNET_FEATURE_PATH_TRAINING'),extractedFeatures,extractedLabels)

    loadedFeatures, loadedLabels = loadLabelled(testDataPath)
    extractedFeatures,extractedLabels = convNetFeatures(loadedFeatures,loadedLabels,model=model)
    saveLabelled(rCon('CONVNET_FEATURE_PATH_TEST'),extractedFeatures,extractedLabels) 