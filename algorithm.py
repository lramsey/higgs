import algorithms.gradientBoosting  as g

def train(method, xTrain, yTrain, tree, metric):
    outputs = []
    if method == 'gboosting':
        outputs = g.train(xTrain, yTrain, tree, metric)
    return outputs

def crossValidate(method, params):
    outputs = []
    if method == 'gboosting':
        outputs = g.crossValidate(*params)
    return outputs

def test(method, tree, params):
    outputs = []
    if method == 'gboosting':
        outputs = g.test(tree, *params)
    return outputs

# explore neural networks
# explore xgboost
