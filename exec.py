import math
import argparse
import numpy     as np
import algorithm as a
import editing   as e

parser = argparse.ArgumentParser()
parser.add_argument('method')
parser.add_argument('scale', nargs='?')
args = parser.parse_args()
method = args.method
scale = args.scale

trainingData = e.readTrainingData()
xTrain = trainingData[0]
yTrain = trainingData[1]
wTrain = trainingData[2]
xValid = trainingData[3]
yValid = trainingData[4]
wValid = trainingData[5]

outputs = a.train(method, scale, [xTrain, yTrain, xValid])

trainClassifications = outputs[0]
validClassifications = outputs[1]

# focus analysis in predicted signal region

scaledTrainWins   = wTrain * (yTrain  == 1.0)*(1.0/0.9)
scaledTrainLosses = wTrain * (yTrain  == 0.0)*(1.0/0.9)
scaledValidWins   = wValid * (yValid  == 1.0)*(1.0/0.1)
scaledValidLosses = wValid * (yValid  == 0.0)*(1.0/0.1)


sTrain  = sum (scaledTrainWins   *(trainClassifications == 1.0))
bTrain  = sum (scaledTrainLosses *(trainClassifications == 1.0))
sValid  = sum (scaledValidWins   *(validClassifications == 1.0))
bValid  = sum (scaledValidLosses *(validClassifications == 1.0))

def AMS(s,b):
    return math.sqrt(2.*((s+b+10.)*math.log(1.+ s/(b+10.) )-s ))

trainScore = AMS(sTrain,bTrain)
testScore  = AMS(sValid, bValid)
print '90% training score: ' + str(trainScore)
print '10% testing score: ' + str(testScore)

n = sValid + bValid

def zScore(n,b):
    return math.sqrt( 2 * (n * math.log(n/b)-n + b))

Z = zScore(n, bValid)

print 'Z: ' + str(Z)

testData  = e.readTestData()
xTest     = testData[0]
indexTest = testData[1]

results = a.test(method, scale, xTest)
testClassifications = results[0]
testResults = list(results[1])


resultList = []
for i in range(len(indexTest)):
    resultList.append([int(indexTest[i]), testResults[i], 's'*(testClassifications[i] == 1.0) + 'b'*(testClassifications[i] == 0.0)])

resultList = sorted(resultList, key=lambda a_entry: a_entry[1])

for i in range(len(resultList)):
    resultList[i][1] = i+1

resultList = sorted(resultList, key=lambda a_entry: a_entry[0])

e.writeTestResults(resultList)
