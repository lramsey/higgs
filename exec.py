import argparse
import algorithm as a
import editing   as e
import numpy     as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('method')
parser.add_argument('scale', nargs='?')
args = parser.parse_args()
method = args.method
scale = args.scale

trainingData = e.readTrainingData()
xTrain       = trainingData[0]
yTrain       = trainingData[1]
wTrain       = trainingData[2]
xCrossValid  = trainingData[3]
yCrossValid  = trainingData[4]
wCrossValid  = trainingData[5]
xTestValid   = trainingData[6]
yTestValid   = trainingData[7]
wTestValid   = trainingData[8]

metric = 91
print 'cross-validation'
# use keyword args
cvScores    = []
trainScores = []
high = [0, -1]
params = [yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid]
methods = ['logistic', 'forest']
for i in range(0, len(methods)):
    method = methods[i]
    a.train(method, xTrain, yTrain, metric)
    score = a.crossValidate(method, params)
    trainScore = score[0]
    cvScore    = score[1]

    trainScores.append(trainScore)
    cvScores.append(cvScore)
    if cvScore > high[0]:
        high = [cvScore, method]

print 'winrar:' + str(high)
winrar = high[1]
params =[xTestValid, yTestValid, wTestValid]
a.train(method, xTrain, yTrain, winrar, metric)
test = a.test(method, params)

# plt.plot()
# plt.plot()
# plt.show()

















# testData  = e.readTestData()
# xTest     = testData[0]
# indexTest = testData[1]

# results = a.test(method, scale, xTest)
# testClassifications = results[0]
# testResults = list(results[1])


# resultList = []
# for i in range(len(indexTest)):
#     resultList.append([int(indexTest[i]), testResults[i], 's'*(testClassifications[i] == 1.0) + 'b'*(testClassifications[i] == 0.0)])

# resultList = sorted(resultList, key=lambda a_entry: a_entry[1])

# for i in range(len(resultList)):
#     resultList[i][1] = i+1

# resultList = sorted(resultList, key=lambda a_entry: a_entry[0])

# e.writeTestResults(resultList)
