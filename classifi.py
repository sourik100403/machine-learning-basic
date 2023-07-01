#loading
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading data sets
iris=datasets.load_iris();
# print(iris.DESCR);


features=iris.data;
label=iris.target;
# print(features[0],label[0]);



#training the classifier
clf=KNeighborsClassifier();
clf.fit(features,label);
preds=clf.predict([[6.5, 3.5, 1.4 ,0.2]])
print(preds)