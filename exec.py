# import math
import argparse
import algorithm as a
import editing   as e
# from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('method')
parser.add_argument('scale', nargs='?')
args = parser.parse_args()
method = args.method
# methods = [method]
# methods = ast.literal_eval(args.methods)
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

params = [xTrain, yTrain, wTrain, xCrossValid, yCrossValid, wCrossValid]
print a.train(method, scale, params)

# trainClassifications = outputs[0]
# crossValidClassifications = outputs[1]
# print classification_report(yTrain, trainClassifications)
# print classification_report(yCrossValid, crossValidClassifications)


# # focus analysis in predicted signal region

# scaledTrainWins   = wTrain * (yTrain  == 1.0)*(1.0/0.6)
# scaledTrainLosses = wTrain * (yTrain  == 0.0)*(1.0/0.6)
# scaledCrossValidWins   = wCrossValid * (yCrossValid  == 1.0)*(1.0/0.2)
# scaledCrossValidLosses = wCrossValid * (yCrossValid  == 0.0)*(1.0/0.2)

# sTrain  = sum (scaledTrainWins   *(trainClassifications == 1.0))
# bTrain  = sum (scaledTrainLosses *(trainClassifications == 1.0))
# sCrossValid  = sum (scaledCrossValidWins   *(crossValidClassifications == 1.0))
# bCrossValid  = sum (scaledCrossValidLosses *(crossValidClassifications == 1.0))

# def AMS(s,b):
#     return math.sqrt(2.*((s+b+10.)*math.log(1.+ s/(b+10.) )-s ))

# trainScore = AMS(sTrain,bTrain)
# testScore  = AMS(sCrossValid, bCrossValid)
# print '90% training score: ' + str(trainScore)
# print '10% testing score: ' + str(testScore)


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
