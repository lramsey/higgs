from sklearn import ensemble as e

forest  = None
forest1 = None
forest2 = None
def randomForest(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid  = params[2]
    outputs = []
    global forest
    forest = e.RandomForestClassifier(n_estimators=300, criterion='entropy', oob_score=True, n_jobs=-1)
    forest.fit(xTrain, yTrain)
    outputs.append(forest.predict_proba(xTrain)[:,1])
    outputs.append(forest.predict_proba(xValid)[:,1])
    return outputs

def test(xTest):
    if forest1 != None:
        return (forest1.predict_proba(xTest)[:,1] + forest2.predict_proba(xTest)[:,1])/2
    else:
        return forest.predict_proba(xTest)[:,1]

def randomForestCombination(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid  = params[2]
    global forest1
    forest1 = e.RandomForestClassifier(n_estimators=150, criterion='entropy', oob_score=True, n_jobs=-1)
    forest1.fit(xTrain, yTrain)
    global forest2
    forest2 = e.RandomForestClassifier(n_estimators=150, criterion='entropy', oob_score=True, n_jobs=-1)
    forest2.fit(xTrain, yTrain)

    forestTrain = (forest1.predict_proba(xTrain)[:,1] + forest2.predict_proba(xTrain)[:,1])/2
    forestValid = (forest1.predict_proba(xValid)[:,1] + forest2.predict_proba(xValid)[:,1])/2
    return [forestTrain, forestValid]
