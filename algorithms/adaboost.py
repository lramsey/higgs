import __init__ as i
from sklearn import ensemble as e

boost = None

def train(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]
    global boost
    boost = e.AdaBoostClassifier(n_estimators=400)
    boost.fit(xTrain, yTrain)
    trainResults = boost.predict_proba(xTrain)[:,1]
    validResults = boost.predict_proba(xValid)[:,1]

    i.setSuccess(trainResults)
    return i.formatOutputs([trainResults, validResults])

def test(xTest):
    testResults = boost.predict_proba(xTest)[:,1]
    testClassification = i.formatOutputs([testResults])[0]
    return [testClassification, testResults]
