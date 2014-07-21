from sklearn import linear_model as lm

logistic = None

def logistic(params, norm=1):
    xTrain = params[0]
    yTrain = params[1]
    xValid  = params[2]
    outputs = []
    global logistic
    logistic = lm.LogisticRegression(C=norm)
    logistic.fit(xTrain,yTrain)
    outputs.append(logistic.predict_proba(xTrain)[:,1])
    outputs.append(logistic.predict_proba(xValid)[:,1])
    return outputs

def test(xTest):
    return logistic.predict_proba(xTest)[:,1]