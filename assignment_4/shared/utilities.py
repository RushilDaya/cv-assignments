import numpy as np
import copy
import pickle

def saveLabelled(path, dataArray, labelArray):
    _dArray = copy.deepcopy(dataArray)
    _lArray = copy.deepcopy(labelArray)

    saveObj = {'data': _dArray, 'labels': _lArray}
    
    with open(path, 'wb') as ofile:
        pickle.dump(saveObj, ofile)
    

def loadLabelled(path):
    
    with open(path, 'rb') as ifile:
        saveObj = pickle.load(ifile)

    return saveObj['data'], saveObj['labels']