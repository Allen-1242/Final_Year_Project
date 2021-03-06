import pandas as pd
import scipy.sparse as sps
path = 'dataset_clean2.csv'
dis = pd.read_csv(path, header=None, names=['Disease', 'Symptom','Disease_num'])

# examine the shape
print(dis.shape)

# examine the first 15 rows
print(dis.head(15))

X = dis.Symptom
y = dis.Disease_num
print(X.shape)
print(y.shape)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(y_train))
print(y_train.values)
y_train_m = sps.csr_matrix(y_train)
print(type(y_train_m))
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# learn the 'vocabulary' of the training data (occurs in-place)
x = vect.fit(X_train.values.astype('U'))
print(x)

print(vect.get_feature_names())

simple_train_dtm = vect.transform(X_train.values.astype('U'))
print(simple_train_dtm)

#print(pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names()))
from sklearn.neighbors import KNeighborsClassifier
# instantiate the model (with the default parameters)
knn = KNeighborsClassifier(n_neighbors=147)

# fit the model with data (occurs in-place)
knn.fit(simple_train_dtm, y_train)

xxyy = X_test
#xxyy = ['uncoordination pleuritic pain snuffle throat sore malaise debilitation dysuria'] 
simple_test_dtm = vect.transform(xxyy.values.astype('U'))
#simple_test_dtm = vect.transform(xxyy)
print(type(simple_test_dtm))


# examine the vocabulary and document-term matrix together
#print(pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names()))

aa=knn.predict(simple_test_dtm)
print(aa)
from sklearn import metrics
print("-----Accuracy-----")
print(metrics.accuracy_score(y_test,aa))
