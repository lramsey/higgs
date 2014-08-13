from sklearn.ensemble import GradientBoostingClassifier as GBC
import __init__ as i

gboost       = None
trainResults = None

def train(xTrain, yTrain, metric):
    print 'goosting'
    global gboost
    gboost = GBC()
    gboost.fit(xTrain,yTrain)
    global trainResults
    trainResults = gboost.predict_proba(xTrain)[:,1]
    i.setSuccess(trainResults, metric)

def crossValidate(yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid):
    crossValidResults = gboost.predict_proba(xCrossValid)[:,1]
    results = i.formatOutputs([trainResults, crossValidResults])
    trainClassifications = results[0]
    crossValidClassifications = results[1]
    return i.cvTrainScores(yTrain, wTrain, trainClassifications, yCrossValid, wCrossValid, crossValidClassifications)

def test(xTest, yTest, wTest):
    testResults = gboost.predict_proba(xTest)[:,1]
    testClassifications = i.formatOutputs([testResults])[0]    
    return i.testScore(yTest, wTest, testClassifications)

