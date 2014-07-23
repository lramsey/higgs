from sklearn import svm as s

vector = None

def train(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]

    global vector
    vector = s.SVC()
    vector.fit(xTrain, yTrain)
    trainResults = vector.predict(xTrain)
    testResults  = vector.predict(xValid)
    return [trainResults, testResults]

def test(xTest):
    testClassification = vector.predict(xTest)
    return [testClassification, testClassification]
