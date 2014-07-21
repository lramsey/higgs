import algorithms.logistic     as log
import algorithms.quadratic as quad
import algorithms.randomForest as f
def train(method, scale, params):
    outputs = []
    if scale == 'quadratic':
        params = quad.scale(params)
    if method == 'logistic':
        outputs = log.logistic(params)
    elif method == 'forest':
        if scale == 'combination':
            outputs = f.randomForestCombination(params)
        else:
            outputs = f.randomForest(params)
    return outputs

def test(method, scale, xTest):
    outputs = []
    if method == 'logistic':
        outputs = log.test(xTest)
    elif method == 'quadratic':
        outputs = quad.test(xTest)
    elif method == 'forest':
        outputs = f.test(xTest)
    return outputs

# explore neural networks
# explore xgboost
