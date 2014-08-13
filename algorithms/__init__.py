import math
import numpy as np
from sklearn.metrics import classification_report

success = None

def setSuccess(train, metric):
    global success
    success = np.percentile(train, metric)

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

def cost(classifications, y):
    squaredDiff = (classifications-y)**2
    cost = np.sum(squaredDiff)/len(y)
    return cost

def precision(classifications, y):
    truePositive = 0.
    positive     = 0.
    for i in range(0,len(y)):
        if classifications[i]:
            positive += 1
            if y[i]:
                truePositive += 1
    return truePositive/positive

def cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications):
    trainScore    = score(yTrain, wTrain, trainClassifications, 0.4)
    cvScore       = score(yCrossValid, wCrossValid, crossValidClassifications, 0.15)
    print classification_report(yTrain, trainClassifications)
    print classification_report(yCrossValid, crossValidClassifications)
    trainCost      = cost(trainClassifications, yTrain)
    cvCost         = cost(crossValidClassifications, yCrossValid)
    trainPrecision = precision(trainClassifications, yTrain)
    cvPrecision    = precision(crossValidClassifications, yCrossValid)
    print '40% training score: ' + str(trainScore)
    print 'train cost: ' + str(trainCost)
    print 'train precision: ' + str(trainPrecision)
    print '15% cross-validation score: ' + str(cvScore)
    print 'cross-validation cost: ' + str(cvCost)
    print 'cross-validation precision: ' + str(cvPrecision)
    return [trainScore, cvScore]

def trainScore(y, w, classifications):
    trainScore = score(y, w, classifications, 0.2)
    print '80% trainScore: ' + str(trainScore)
    
def testScore(yTest, wTest, testClassifications):
    testScore = score(yTest, wTest, testClassifications, 0.2)
    testCost       = cost(testClassifications, yTest)
    testPrecision  = precision(testClassifications, yTest)
    print classification_report(yTest, testClassifications)
    print '15% test score: ' + str(testScore)
    print 'test cost: ' + str(testCost)
    print 'test precision: ' + str(testPrecision)
    return testScore

def AMS(s,b):
    return math.sqrt(2.*((s+b+10.)*math.log(1.+ s/(b+10.) )-s ))
