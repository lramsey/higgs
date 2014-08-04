from sklearn.neural_network import BernoulliRBM as Bernoulli

neural = None

def scale(params):
    xTrain = params[0]
    yTrain = params[1]
    xValid = params[2]
    print 'neuralizing'
    global neural
    neural = Bernoulli()
    xTrain = neural.fit_transform(xTrain, yTrain)
    xValid = neural.transform(xValid)

    return [xTrain, yTrain, xValid]

def testScale(xTest):
    global neural
    neural = Bernoulli()
    xTest = neural.transform(xTest)
    return xTest
