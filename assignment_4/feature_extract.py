import numpy as np 
from shared.configurationParser import retrieveConfiguration as rCon 
from shared.utilities import loadLabelled, saveLabelled
from shared.histogramFeatureMethods import histogramFeatures

trainingDataPath = rCon('DATA_NAME_TRAINING')
testDataPath = rCon('DATA_NAME_TEST')


if rCon('RUN_HISTOGRAM_FEATURE_EXTRACT'):
    loadedFeatures, loadedLabels = loadLabelled(trainingDataPath)
    extractedFeatures,extractedLabels = histogramFeatures(loadedFeatures,loadedLabels,bucket_resolution=rCon('HISTOGRAM_BUCKET_RESOLUTION'))
    saveLabelled(rCon('HISTOGRAM_FEATURE_PATH_TRAINING'),extractedFeatures,extractedLabels)

    loadedFeatures, loadedLabels = loadLabelled(testDataPath)
    extractedFeatures,extractedLabels = histogramFeatures(loadedFeatures,loadedLabels,bucket_resolution=rCon('HISTOGRAM_BUCKET_RESOLUTION'))
    saveLabelled(rCon('HISTOGRAM_FEATURE_PATH_TEST'),extractedFeatures,extractedLabels)
