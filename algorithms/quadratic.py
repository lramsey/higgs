import numpy as np

def scale(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid  = params[2]
    length = len(yTrain)
    testLength = len(xValid)
    terms = termBuilder(xTrain[0])
    count = len(terms)

    xTrain = featureBuilder(xTrain, terms, count, length)
    xValid  = featureBuilder(xValid, terms, count, testLength)
    
    return [xTrain, yTrain, xValid]

def termBuilder(sample):
    terms = []
    for i in range(0,len(sample)):
        terms.append([i])
        for j in range (i, len(sample)):
            terms.append([i,j])
    return terms

def featureBuilder(X, terms, count, length):
    newX = np.zeros((length, count))
    for i in range(0,count):
        p = np.zeros(length)
        p = p + X[:,terms[i][0]]
        if len(terms[i]) > 1:
            p *= X[:,terms[i][1]]
        newX[:,i] = p
    return newX