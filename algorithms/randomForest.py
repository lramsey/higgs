from sklearn.ensemble import RandomForestClassifier
import __init__ as i

forest       = None
trainResults = None

def train(xTrain, yTrain, metric):
    print 'RandomForestClassifier'
    global forest
    forest = RandomForestClassifier()
    forest.fit(xTrain, yTrain)
    global trainResults
    trainResults = forest.predict(xTrain)
    i.setSuccess(trainResults, metric)

def crossValidate(yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid):
    crossValidResults = forest.predict_proba(xCrossValid)[:,1]
    results = i.formatOutputs([trainResults, crossValidResults])
    trainClassifications = results[0]
    crossValidClassifications = results[1]
    return i.cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications)

def test(xTest, yTest, wTest):
    testResults = forest.predict_proba(xTest)[:,1]
    testClassifications = i.formatOutputs([testResults])[0]
    return i.testScore(yTest, wTest, testClassifications)
