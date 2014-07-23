from sklearn import ensemble as e
import __init__ as i

gboost = None

def train(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]
    global gboost
    print 'n_est: 50'
    print 'max_depth: 10'
    print 'min_samples_leaf: 200'
    print 'max_features: 10'
    gboost = e.GradientBoostingClassifier(n_estimators=50, max_depth=10,min_samples_leaf=200, max_features=10)
    gboost.fit(xTrain,yTrain)
    trainResults = gboost.predict_proba(xTrain)[:,1]
    validResults = gboost.predict_proba(xValid)[:,1]

    i.setSuccess(trainResults)
    return i.formatOutputs([trainResults, validResults])

def test(xTest):
    testResults = gboost.predict_proba(xTest)[:,1]
    testClassification = i.formatOutputs([testResults])[0]
    return [testClassification, testResults]
