import sys
import math
from operator import itemgetter
from collections import OrderedDict
import pandas as pd


def gaussianDistr(x,mean,var):
	sd = float(math.sqrt(var))
	numerator = math.exp(float(-(1.0/2)) * (float(((float(x)-float(mean))**2)/float(var))))
	denominator = float(sd) * math.sqrt(float(2.0) * math.pi)
	return float(numerator)/float(denominator)

def fetchAccuracy(data,colName,resultDict):
	classDict = {}
	for index in data.index:
		key = data.iloc[index]["ID"]
		classDict.setdefault(key,0)
		maxProb = 0.0
		assignedClass = 0
		for classNo in range(1,8):
			prob = resultDict[classNo]["PRIOR"]
			for attrNo,attr in enumerate(colName):
				if attr not in ("ID","CLASS"):
					x = data.iloc[index][attr]
					mean = resultDict[classNo]["MEAN"][attrNo-1]
					var =  resultDict[classNo]["VAR"][attrNo-1]
					if var!=0:
						prob = prob * gaussianDistr(x,mean,var)
					elif x == mean:
						prob = prob * 1.0
					else:
						prob = prob * 0.0

			if prob >= maxProb:
				maxProb = prob
				assignedClass = classNo

		classDict[key] = assignedClass

	count = 0
	for key in classDict.keys():
		if data.loc[data["ID"] == key].iloc[0]["CLASS"] == classDict[key] :
			count = count+1
	return float(count)/data["ID"].count()

def manhattan(list1,list2,meanForNorm,sdForNorm):
	sum = float(0)
	for i in range(len(list1)):
		sum = sum + math.fabs(float((float(list1[i]) - float(meanForNorm[i]))/float(sdForNorm[i])) - float((float(list2[i]) - float(meanForNorm[i]))/float(sdForNorm[i])))
	return float(sum)


def euclidean(list1,list2,meanForNorm,sdForNorm):
	sum = float(0)
	for i in range(len(list1)):
		sum = sum + ((float((float(list1[i]) - float(meanForNorm[i]))/float(sdForNorm[i])) - float((float(list2[i]) - float(meanForNorm[i]))/float(sdForNorm[i])))**2)
	return float(math.sqrt(sum))

def kNN(k,distance,testData,trainData,colName,accuracyType,meanForNorm,sdForNorm):
	distDict= OrderedDict()
	# Testing Accuracy
	if accuracyType == "TEST":
		for index in testData.index:
			testKey = testData.iloc[index]["ID"]
			distDict.setdefault(testKey,OrderedDict())
			for i in trainData.index:
				trainKey = trainData.iloc[i]["ID"]
				distDict[testKey].setdefault(trainKey,[0]*2)
				testList = testData.iloc[index][testData.columns[1:10]].tolist()
				trainList = trainData.iloc[i][trainData.columns[1:10]].tolist()
				if distance == "L1": #L1 = Manhattan Distance
					distDict[testKey][trainKey][0]=manhattan(testList,trainList,meanForNorm,sdForNorm)
				elif distance == "L2": #L2 = Eucledian Distance
					distDict[testKey][trainKey][0]=euclidean(testList,trainList,meanForNorm,sdForNorm)
				distDict[testKey][trainKey][1]=trainData.iloc[i]["CLASS"]

	elif accuracyType == "TRAIN":
		for index in trainData.index:
			testKey = trainData.iloc[index]["ID"]
			distDict.setdefault(testKey,OrderedDict())
			for i in trainData.index:
				if i != index :
					trainKey = trainData.iloc[i]["ID"]
					distDict[testKey].setdefault(trainKey,[0]*2)
					testList = trainData.iloc[index][trainData.columns[1:10]].tolist()
					trainList = trainData.iloc[i][trainData.columns[1:10]].tolist()
					if distance == "L1": #L1 = Manhattan Distance
						man = manhattan(testList,trainList,meanForNorm,sdForNorm)
						if man == 0:
							distDict[testKey][trainKey][0] = float('inf')
						else:
							distDict[testKey][trainKey][0] = man
					elif distance == "L2": #L2 = Eucledian Distance
						euc = euclidean(testList,trainList,meanForNorm,sdForNorm)
						if euc == 0:
							distDict[testKey][trainKey][0] = float('inf')
						else:
							distDict[testKey][trainKey][0] = euc
						
					distDict[testKey][trainKey][1]=trainData.iloc[i]["CLASS"]


	resultDict = {}
	for dictKey in distDict.keys():
		resultDict.setdefault(dictKey,0)
		kNN = OrderedDict(sorted(distDict[dictKey].items(), key= lambda x:x[1])[0:int(k)])
		classCount = [0]*7
		for key in kNN.keys():
			classCount[int(kNN[key][1])-1] = classCount[int(kNN[key][1])-1] + 1
			maxCount = max(classCount)
			maxCountClass = [i+1 for i,j in enumerate(classCount) if j == maxCount]
			for key in kNN.keys():
				if int(kNN[key][1]) in maxCountClass:
					resultDict[dictKey] = kNN[key][1]
					break

	count = 0
	totalCount = 0
	if accuracyType == "TEST":
		totalCount = testData["ID"].count()

		for key in resultDict.keys():
			if testData.loc[testData["ID"] == key].iloc[0]["CLASS"] == resultDict[key] :
				count = count+1
	elif accuracyType == "TRAIN":
		totalCount = trainData["ID"].count()

		for key in resultDict.keys():
			if trainData.loc[trainData["ID"] == key].iloc[0]["CLASS"] == resultDict[key] :
				count = count+1

	return (float(count)/float(totalCount))*100.0


if __name__ == "__main__":

	##### READING TRAINING FILE ###########
	trainingFile = "train.txt"
	colName = ["ID","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","CLASS"]
	trainData = pd.read_csv(trainingFile, header=None, names= colName)

	##### READING TEST FILE ###########
	testingFile = "test.txt"
	testData = pd.read_csv(testingFile, header=None, names=colName)

	######   NAIVE BAYES  ###############
	resultDict = {}
	for i in range(1,8):
		resultDict.setdefault(i,{})
		resultDict[i]["PRIOR"] = trainData.loc[trainData["CLASS"] == i]["CLASS"].count()*1.0/trainData["ID"].count()
		resultDict[i].setdefault("MEAN",[0]*9)
		resultDict[i].setdefault("VAR",[0]*9)
		for attrNo,attr in enumerate(colName):
			if attr not in ("ID","CLASS"):
				if not trainData.loc[trainData["CLASS"] == i].empty:
					resultDict[i]["MEAN"][attrNo-1] = trainData.loc[trainData["CLASS"] == i][attr].mean()
					resultDict[i]["VAR"][attrNo-1] = trainData.loc[trainData["CLASS"] == i][attr].var()

	print "::::::::::::::::: NAIVE BAYES ::::::::::::::::::::"
	testAccuracy = fetchAccuracy(testData,colName,resultDict)
	print "Testing Accuracy:" + str(testAccuracy*100.0)

	trainAccuracy = fetchAccuracy(trainData,colName,resultDict)
	print "Training Accuracy:" + str(trainAccuracy*100.0)
	print ""

	###############   KNN  ##################

	print "::::::::::::::::: KNN ::::::::::::::::::::"

	meanForNorm = [0]*9
	sdForNorm = [0]*9
	for attrNo,attr in enumerate(colName):
		if attr not in ("ID","CLASS"):
			meanForNorm[attrNo-1] = float(trainData[attr].mean())
			sdForNorm[attrNo-1] = float(trainData[attr].std())

	print "Testing Accuracy:"
	for i in [1,3,5,7]:
		print "k = " + str(i) + " and distance = Manhattan(L1) :" + str(kNN(i,"L1",testData,trainData,colName,"TEST",meanForNorm,sdForNorm))
	for i in [1,3,5,7]:
		print "k = " + str(i) + " and distance = Euclidean(L2) :" + str(kNN(i,"L2",testData,trainData,colName,"TEST",meanForNorm,sdForNorm))

	print "Training Accuracy:"
	for i in [1,3,5,7]:
		print "k = " + str(i) + " and distance = Manhattan(L1) :" + str(kNN(i,"L1",testData,trainData,colName,"TRAIN",meanForNorm,sdForNorm))
	for i in [1,3,5,7]:
		print "k = " + str(i) + " and distance = Euclidean(L2) :" + str(kNN(i,"L2",testData,trainData,colName,"TRAIN",meanForNorm,sdForNorm))
