import numpy as np

success = None

def setSuccess(train):
    global success
    success = np.percentile(train, 85)

def formatOutputs(arr):
    length = len(arr)
    results = []
    for i in range(0,length):
        results.append(arr[i] > success)
    return results

def savePredictions(train, valid, method, num):
    np.save(np.array([train, valid]), 'caches/' + method + str(num))

def loadPredictions(method, num):
    arrays = np.load('caches/' + method + str(num))
    print 'arrays stored:' + str(len(arrays))