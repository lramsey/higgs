import numpy as np
from sklearn.preprocessing import MinMaxScaler as MMS

def readTrainingData():
    data = np.loadtxt( 'data/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) })
    allY = data[:, 32]
    allX = data[:, 1:31]
    allW = data[:, 31]
    
    scale = MMS()
    allX = scale.fit_transform(allX)
    np.random.seed(42)
    r = np.random.rand(allY.shape[0])

    xTrain = allX[r<=0.4]
    yTrain = allY[r<=0.4]
    wTrain = allW[r<=0.4]

    xValid = allX[r>0.7]
    yValid = allY[r>0.7]
    wValid = allW[r>0.7]

    v = np.random.rand(yValid.shape[0])
    xCrossValid = xValid[v<=0.5]
    yCrossValid = yValid[v<=0.5]
    wCrossValid = wValid[v<=0.5]

    xTestValid  = xValid[v>0.5]
    yTestValid  = yValid[v>0.5]
    wTestValid  = wValid[v>0.5]
    
    return [xTrain, yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid, xTestValid, yTestValid, wTestValid]


def readTestData():
    testData  = np.loadtxt('data/test.csv', delimiter=',', skiprows=1)
    xTest     = testData[:,1:31]
    scale = MMS()
    allX = scale.fit_transform(xTest)
    indexTest = list(testData[:,0])
    return [allX, indexTest]

def writeTestResults(resultList):
    fcsv = open('results/Kaggle_higgs_prediction_outputGrad50.csv','w')
    fcsv.write('EventId,RankOrder,Class\n')
    for line in resultList:
        theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
        fcsv.write(theline) 
    fcsv.close()
