import algorithms.logistic          as log
import algorithms.quadratic         as quad
import algorithms.randomForest      as f
import algorithms.svm               as s
import algorithms.adaboost          as ada
import algorithms.gradientBoosting  as g

def train(method, scale, params):
    outputs = []
    if scale == 'quadratic':
        params = quad.scale(params)
    if method == 'logistic':
        outputs = log.train(params)
    elif method == 'forest':
        outputs = f.train(params)
    elif method == 'combforest':
        outputs = f.randomForestCombination(params)
    elif method == 'svm':
        outputs = s.train(params)
    elif method == 'adaboost':
        outputs = ada.train(params)
    elif method == 'gboosting':
        outputs = g.train(params)
    return outputs

def test(method, scale, xTest):
    outputs = []
    if scale == 'quadratic':
        xTest = quad.testScale(xTest)
    if method == 'logistic':
        outputs = log.test(xTest)
    elif method == 'forest':
        outputs = f.test(xTest)
    elif method == 'svm':
        outputs = s.test(xTest)
    elif method == 'adaboost':
        outputs = ada.test(xTest)
    elif method == 'gboosting':
        outputs = g.test(xTest)
    return outputs

# explore neural networks
# explore xgboost
