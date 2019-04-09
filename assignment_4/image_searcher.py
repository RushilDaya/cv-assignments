''' 
    the primary output of the project
    this script performs the actual image search
'''

from shared.configurationParser import retrieveConfiguration as rCon 
from shared.histogramFeatureMethods import singleImageHisto
from shared.convnetFeatureMethods import singleImageConv
from shared.localBinaryPatternsMethods import singleImageLBP
from shared.similarityMethods import getMostLeastSimilar
from shared.utilities import loadLabelled, loadObj
import matplotlib.pyplot as plt

playIndex = rCon('PLAY_IMAGE_INDEX')
playImages, playLabels = loadLabelled(rCon('DATA_NAME_PLAY'))
selectedImage = playImages[playIndex]
selectedImageClass = playLabels[playIndex]


method = rCon('SEARCHER_METHOD')
if method == 'HISTOGRAM':
    dbFeatures, dbLabels = loadLabelled(rCon('HISTOGRAM_FEATURE_PATH_TRAINING'))
    features = singleImageHisto(selectedImage, bucket_resolution=rCon('HISTOGRAM_BUCKET_RESOLUTION'))


elif method == 'CNN':
    dbFeatures, dbLabels = loadLabelled(rCon('CONVNET_FEATURE_PATH_TRAINING'))
    model = loadObj(rCon('CONVNET_MODEL_PATH'))
    features = singleImageConv(selectedImage, model=model)

elif method == 'LBP':
    dbFeatures, dbLabels = loadLabelled(rCon('LBP_FEATURE_PATH_TRAINING'))
    features = singleImageLBP(selectedImage)

else:
    raise TypeError()

msIdx, lsIdx, _, _ = getMostLeastSimilar(dbFeatures, features, qty_most=4 , qty_least=4)
dbImages,_ = loadLabelled(rCon('DATA_NAME_TRAINING'))

plt.imshow(selectedImage)
plt.show()

mostSimilarImages = dbImages[msIdx].tolist()
for image in mostSimilarImages:
    plt.imshow(image)
    plt.show()

