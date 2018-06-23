import pandas as pd
import numpy as np
import operator 
path = 'dataset_clean2.csv'
disease_file = pd.read_csv(path, header=None, names=['Disease', 'Symptom','Disease_num'])

# examine the shape
print(disease_file.shape)
# examine the first 10 rows
print(disease_file.head(10))
# examine the class distribution
print(disease_file.Disease.value_counts())

X = disease_file.Symptom
y = disease_file.Disease_num
print(X.shape)
print(y.shape)


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

print(vect.fit(X_train.values.astype('U')))
X_train_dtm = vect.transform(X_train.values.astype('U'))
print('X_train_dtm type = ')
print(type(X_train_dtm))
#print(X_train_dtm)
#arrdtm=np.array(X_train_dtm)
print('array train dtm')
#c,r = arrdtm.shape
#arrdtm=arrdtm.reshape(c,)
#print(arrdtm)
#X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test.values.astype('U'))
#print(X_test_dtm)
print('X_test_dtm type = ')
print(type(X_test_dtm.dtype))
#arrdtmtest=np.array(X_test_dtm)
print('array test dtm')
#co,ro = arrdtmtest.shape
#arrdtmtest=arrdtmtest.reshape(c,)
#print(arrdtmtest)

#calculate Euclidean distance in PythonPython
import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(1,length+1):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][0] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data

	print ('Train set: ' + repr(len(X_train)))
	print ('Test set: ' + repr(len(X_test)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(X_test)):
		neighbors = getNeighbors(arrdtm, arrdtmtest[x], k)
	#	result = getResponse(neighbors)
	#	predictions.append(result)
	#	print('> predicted=' + repr(result) + ', actual=' + repr(arrdtmtest[x][-1]))
	#accuracy = getAccuracy(arrdtmtest, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')
	
main()


#data1 = ['a', 2, 2]
#data2 = ['b', 4, 4]
#distance = euclideanDistance(data1, data2, 1)
#print ('Distance: ' + repr(distance))


#trainSet = [['a', 2, 2], ['b', 4, 4]]
#testInstance = [5, 5]
#k = 1
#neighbors = getNeighbors(trainSet, testInstance, 1)
#print(neighbors)


#neighbors = [['a', 2, 2], ['b', 4, 4], ['b',5,5]]
#print('nei = ' + repr(len(neighbors)))
#response = getResponse(neighbors)
#print(response)



#testSet = [['a',2,2], ['a',3,3], ['b',4,4]]
#predictions = ['a', 'a', 'a']
#accuracy = getAccuracy(testSet, predictions)
#print(accuracy)
