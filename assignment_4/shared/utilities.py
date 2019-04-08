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

def saveObj(path, ob):
    with open(path,'wb') as ofile:
        pickle.dump(ob, ofile)


def loadObj(path):
    with open(path,'rb') as ifile:
        ret = pickle.load(ifile)
    return ret

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()