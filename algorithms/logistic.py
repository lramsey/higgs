from sklearn import linear_model as lm
import __init__ as i

logis = None

def train(params, norm=1):
    xTrain = params[0]
    yTrain = params[1]
    xValid  = params[2]
    global logis
    logis = lm.LogisticRegression(C=norm)
    logis.fit(xTrain,yTrain)
    trainResults = logis.predict_proba(xTrain)[:,1]
    validResults = logis.predict_proba(xValid)[:,1]

    i.setSuccess(trainResults)
    return i.formatOutputs([trainResults, validResults])

def test(xTest):
    testResults = logis.predict_proba(xTest)[:,1]
    testClassification = i.formatOutputs([testResults])[0]
    return [testClassification, testResults]
