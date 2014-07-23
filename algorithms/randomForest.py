import __init__ as i
from sklearn import ensemble as e

forest       = None
forestComb   = False

def train(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]
    global forest
    forest = e.RandomForestClassifier(n_estimators=100, criterion='entropy', oob_score=True, n_jobs=-1)
    forest.fit(xTrain, yTrain)

    trainResults = forest.predict_proba(xTrain)[:,1]
    validResults = forest.predict_proba(xValid)[:,1]

    i.savePredictions(trainResults, validResults, 'forest', 100)
    i.loadPredictions('forest', 100)
    i.setSuccess(trainResults)    
    return i.formatOutputs([trainResults, validResults])

def test(xTest):
    testResults = forest.predict_proba(xTest)[:,1]
    testClassifications = i.formatOutputs([testResults])[0]
    return [testClassifications, testResults]

# def testComb(xTest):
#     testResults = (forest1.predict_proba(xTest)[:,1] + forest2.predict_proba(xTest)[:,1])/2

# def randomForestCombination(params):
#     xTrain = params[0]
#     yTrain = params[1]
#     xValid = params[2]

#     global forestComb
#     forestComb = True


#     global forest2

#     trainResults = (forest1.predict_proba(xTrain)[:,1] + forest2.predict_proba(xTrain)[:,1])/2
#     validResults = (forest1.predict_proba(xValid)[:,1] + forest2.predict_proba(xValid)[:,1])/2

#     i.setSuccess(trainResults)
#     return i.formatOutputs([trainResults, validResults])

