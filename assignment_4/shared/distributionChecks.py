
from shared.configurationParser import retrieveConfiguration as rConfig


def uniformDistribution(vector, numClasses, tolerance=None):
    if tolerance == None:
        tol = rConfig('UNIFORM_DISTRIBUTION_TOLERANCE')
    else:
        tol = tolerance
    
    tally = [0]*numClasses

    for item in vector:
        tally[item] +=1
    
    expectedCount = float(len(vector))/numClasses

    for clss in tally:
        diff = abs(expectedCount - clss)
        if float(diff)/expectedCount > tol:
            return False

    return True
    