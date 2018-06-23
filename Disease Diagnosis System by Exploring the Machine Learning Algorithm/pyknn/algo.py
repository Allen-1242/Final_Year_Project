import pandas as pd
path = 'dataset_clean2.csv'
dis = pd.read_csv(path, header=None, names=['Disease', 'Symptom','Disease_num'])

# examine the shape
print(dis.shape)

# examine the first 10 rows
print(dis.head(20))

X = dis.Symptom
y = dis.Disease_num
print(X.shape)
print(y.shape)
#xx=dis.Symptom
#print(xx.shape)
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

vect.fit(X_train.values.astype('U'))
#print(vect.get_feature_names())
X_train_dtm = vect.transform(X_train.values.astype('U'))
print(X_train_dtm)

X_test_dtm = vect.transform(X_test.values.astype('U'))
print(X_test_dtm)
#vect.fit(xx.values.astype('U'))
#print(vect.get_feature_names())
#xx_train_dtm = vect.transform(xx.values.astype('U'))
#print(xx_train_dtm)
#print(xx_train_dtm.toarray())
#a=pd.DataFrame(xx_train_dtm.toarray(), columns=vect.get_feature_names())
#print(a)
# equivalently: combine fit and transform into a single step
#X_train_dtm = vect.fit_transform(X_train.values.astype('U'))

# examine the document-term matrix
#print(X_train_dtm)

# transform testing data (using fitted vocabulary) into a document-term matrix
#X_test_dtm = vect.transform(X_test.values.astype('U'))
#print(X_test_dtm)

#Euclidean Distance
import math
def euclideanDistance(instance1, instance2, x):
	distance = 0
	distance += pow((instance1[x-1] - instance2[x-1]), 2)
	return math.sqrt(distance)
    
#for testing the above function
#data1 = [2, 2]
#data2 = [6, 4]
#distance = euclideanDistance(data1, data2, 1)
#print ('Distance: ' + repr(distance))

import operator
def getNeighbors(trainingSet, testInstance, k):
	distances = []
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
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0	
def main():
    print('generate predictions')
    predictions=[]
    k = 3
    print('1')
    for x in range(len(y_train)):
        print('2')
        neighbors=getNeighbors(X_train_dtm, y[x],k)
        result=getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(y_train[x][-1]))
    accuracy = getAccuracy(y, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
	
main()
