import algorithms.gradientBoosting  as g
import algorithms.randomForest      as f
import algorithms.logistic          as l
import algorithms.adaboost          as a
import algorithms.svm               as s

def train(method, xTrain, yTrain, metric):
    outputs = []
    if method == 'gboosting':
        outputs = g.train(xTrain, yTrain, metric)
    elif method == 'forest':
        outputs = f.train(xTrain, yTrain, metric)
    elif method == 'logistic':
        outputs = l.train(xTrain, yTrain, metric)
    elif method == 'adaboost':
        outputs = a.train(xTrain, yTrain, metric)
    elif method == 'svm':
        outputs = s.train(xTrain, yTrain, metric)
    return outputs

def crossValidate(method, params):
    outputs = []
    if method == 'gboosting':
        outputs = g.crossValidate(*params)
    elif method == 'forest':
        outputs = f.crossValidate(*params)
    elif method == 'logistic':
        outputs = l.crossValidate(*params)
    elif method == 'adaboost':
        outputs = a.crossValidate(*params)
    elif method == 'svm':
        outputs = s.crossValidate(*params)
    return outputs

def test(method, params):
    outputs = []
    if method == 'gboosting':
        outputs = g.test(*params)
    elif method == 'forest':
        outputs = f.test(*params)
    elif method == 'logistic':
        outputs = l.test(*params)
    elif method == 'adaboost':
        outputs = a.test(*params)
    elif method == 'svm':
        outputs = s.test(*params)
    return outputs
