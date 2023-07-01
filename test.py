import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes();
# diabetes_x=diabetes.data[:,np.newaxis,2];#its work for 2 row convert a one colume 
diabetes_x=np.array([[1],[2],[3]]);#all features available
print(diabetes_x);# its print array of array
diabetes_x_train=diabetes_x;
diabetes_x_test=diabetes_x;
diabetes_y_train=np.array([3,2,4]);
diabetes_y_test=np.array([3,2,4]);

model=linear_model.LinearRegression();
model.fit(diabetes_x_train,diabetes_y_train);#training
diabetes_y_predicted= model.predict(diabetes_x_test);#testing


#mean squre error 
print("mean squere error is :",mean_squared_error(diabetes_y_test,diabetes_y_predicted));
print("weight:",model.coef_);
print("interseft",model.intercept_);

# mean squere error is : 3035.060115291269
# weight: [941.43097333]
# interseft 153.39713623331644

# plot 
plt.scatter(diabetes_x_test,diabetes_y_test);
plt.plot(diabetes_x_test,diabetes_y_predicted);
plt.show();