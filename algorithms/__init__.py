import math
import numpy as np
from sklearn.metrics import classification_report

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

def score(y, w, classifications, ratio):
    scaledWins   = w * (y  == 1.0)*(1.0/ratio)
    scaledLosses = w * (y  == 0.0)*(1.0/ratio)
    
    s  = sum (scaledWins   *(classifications == 1.0))
    b  = sum (scaledLosses *(classifications == 1.0))

    score = AMS(s,b)
    return score

    
def cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications):
    trainScore    = score(yTrain, wTrain, trainClassifications, 0.6)
    cvScore       = score(yCrossValid, wCrossValid, crossValidClassifications, 0.2)
    print classification_report(yTrain, trainClassifications)
    print classification_report(yCrossValid, crossValidClassifications)
    
    print '60% training score: ' + str(trainScore)
    print '20% cross-validation score: ' + str(cvScore)
    return cvScore

def testScores(yTest, wTest, testClassifications):
    testScore = score(yTest, wTest, testClassifications, 0.2)
    print classification_report(yTest, testClassifications)
    print '20% test score: ' + str(testScore)
    return testScore

def AMS(s,b):
    return math.sqrt(2.*((s+b+10.)*math.log(1.+ s/(b+10.) )-s ))
