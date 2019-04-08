''' this script runs precision and recall for  all feature methods
    the decision of chosing a best representation however is not done
    that part is left for the user to decide on'''

import numpy as np 
from shared.utilities import loadLabelled, saveObj
from shared.similarityMethods import normalizeSet, precisionRecallValidation
from shared.configurationParser import retrieveConfiguration as rCon 


#---- deals with the conv net ----------------------------------------------------------------------
trainingFeaturesConv, trainingLabels = loadLabelled(rCon('CONVNET_FEATURE_PATH_TRAINING'))
validationFeaturesConv, validationLabels = loadLabelled(rCon('CONVNET_FEATURE_PATH_TEST'))
validationFeaturesConv = validationFeaturesConv[0:rCon('DATA_NUM_VALIDATION')]
validationLabels = validationLabels[0:rCon('DATA_NUM_VALIDATION')]

trainingFeaturesConvNormed, normFactors = normalizeSet(trainingFeaturesConv)
validationFeaturesConvNormed,_ = normalizeSet(validationFeaturesConv, existing_norm_factors=normFactors)

precisionConv, recallConv = precisionRecallValidation(trainingFeaturesConvNormed, trainingLabels,
                                                        validationFeaturesConvNormed, validationLabels,
                                                        qty_returned=rCon('NUM_SIMILAR_RETURNED'))
#---------------------------------------------------------------------------------------------------------

#---- deals with the color histogram----------------------------------------------------------------------
trainingFeaturesHisto, trainingLabels = loadLabelled(rCon('HISTOGRAM_FEATURE_PATH_TRAINING'))
validationFeaturesHisto, validationLabels = loadLabelled(rCon('HISTOGRAM_FEATURE_PATH_TEST'))
validationFeaturesHisto = validationFeaturesHisto[0:rCon('DATA_NUM_VALIDATION')]
validationLabels = validationLabels[0:rCon('DATA_NUM_VALIDATION')]

trainingFeaturesHistoNormed, normFactors = normalizeSet(trainingFeaturesHisto)
validationFeaturesHistoNormed,_ = normalizeSet(validationFeaturesHisto, existing_norm_factors=normFactors)

precisionHisto, recallHisto = precisionRecallValidation(trainingFeaturesHistoNormed, trainingLabels,
                                                        validationFeaturesHistoNormed, validationLabels,
                                                        qty_returned=rCon('NUM_SIMILAR_RETURNED'))
#---------------------------------------------------------------------------------------------------------



results = {
    'Histogram Precision':precisionHisto,
    'Histogram Recall':recallHisto,
    'Conv Precision':precisionConv,
    'Conv Recall':recallConv,
    'Number Images Returned':rCon('NUM_SIMILAR_RETURNED'),
    'Number of validation Images':rCon('DATA_NUM_VALIDATION')
}

print(results)
saveObj(rCon('VALIDATION_RESULTS_PATH'),results)