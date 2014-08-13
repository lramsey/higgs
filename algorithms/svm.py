from sklearn.svm import SVC
import __init__ as i

vector       = None
trainResults = None

def train(xTrain, yTrain, metric):
    print 'svm'
    global vector
    vector = SVC()
    vector.fit(xTrain, yTrain)
    global trainResults
    trainResults = vector.predict(xTrain)
    i.setSuccess(trainResults, metric)

def crossValidate(yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid):
    crossValidResults = vector.predict_proba(xCrossValid)[:,1]
    results = i.formatOutputs([trainResults, crossValidResults])
    trainClassifications = results[0]
    crossValidClassifications = results[1]
    return i.cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications)

def test(xTest, yTest, wTest):
    testResults = vector.predict_proba(xTest)[:,1]
    testClassifications = i.formatOutputs([testResults])[0]    
    return i.testScore(yTest, wTest, testClassifications)
