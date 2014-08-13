from sklearn.linear_model import LogisticRegression
import __init__ as i

logis        = None
trainResults = None

def train(xTrain, yTrain, metric):
    print 'logistic'
    global logis
    logis = LogisticRegression()
    logis.fit(xTrain,yTrain)
    global trainResults
    trainResults = logis.predict_proba(xTrain)[:,1]
    i.setSuccess(trainResults, metric)

def crossValidate(yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid):
    crossValidResults = logis.predict_proba(xCrossValid)[:,1]
    results = i.formatOutputs([trainResults, crossValidResults])
    trainClassifications = results[0]
    crossValidClassifications = results[1]
    return i.cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications)

def test(xTest, yTest, wTest):
    testResults = logis.predict_proba(xTest)[:,1]
    testClassifications = i.formatOutputs([testResults])[0]    
    return i.testScore(yTest, wTest, testClassifications)
