from sklearn.ensemble import GradientBoostingClassifier as GBC
import __init__ as i

gboost     = None

def train(params):
    xTrain      = params[0]
    yTrain      = params[1]
    wTrain      = params[2]
    xCrossValid = params[3]
    yCrossValid = params[4]
    wCrossValid = params[5]

    global gboost
    gboost = GBC()
    gboost.fit(xTrain,yTrain)
    crossValidate(xTrain, yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid)

def test(xTest):
    testResults = gboost.predict_proba(xTest)[:,1]
    testClassification = i.formatOutputs([testResults])[0]
    return [testClassification, testResults]

def crossValidate(xTrain, yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid):
    trainResults = gboost.predict_proba(xTrain)[:,1]
    crossValidResults = gboost.predict_proba(xCrossValid)[:,1]
    i.setSuccess(trainResults)
    results = i.formatOutputs([trainResults, crossValidResults])
    trainClassifications = results[0]
    crossValidClassifications = results[1]
    return i.cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications)
