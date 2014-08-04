from sklearn.ensemble import GradientBoostingClassifier as GBC
import numpy as np
import __init__ as i

gboost     = None

def train(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]

    print 'n_est: 50'
    print 'max_depth: 10'
    print 'min_samples_leaf: 200'
    print 'max_features: 10'

    global gboost
    gboost = GBC(n_estimators=800, max_depth=8,min_samples_leaf=100, max_features='auto')
    gboost.fit(xTrain,yTrain)
    trainResults = gboost.predict_proba(xTrain)[:,1]
    validResults = gboost.predict_proba(xValid)[:,1]

    i.setSuccess(trainResults)
    return i.formatOutputs([trainResults, validResults])

def test(xTest):
    testResults = gboost.predict_proba(xTest)[:,1]
    testClassification = i.formatOutputs([testResults])[0]
    return [testClassification, testResults]

def predict(train, valid):
    trainResults = gboost.predict_proba(train)[:,1]
    validResults = gboost.predict_proba(valid)[:,1]
    i.setSuccess(trainResults)
    return i.formatOutputs([trainResults, validResults])