#train a logistic regression classifier topredict weather a iris verginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

#load data
iris=datasets.load_iris()
x=iris["data"][:,3:]
y=(iris["target"] ==2).astype(np.int)
# print(x)
# print(y)


#train a logictic regression classifier
clf=LogisticRegression();
clf.fit(x,y)

#predict the data
example=clf.predict(([[2.6]]))
print(example)

#using matplotlib graph
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new,y_prob[:,1],"g-",label="VIRGINICIA")
plt.show()



# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])