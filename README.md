# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: POPURI SRAVANI
RegisterNumber: 2122232440117
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv("student_scores.csv")

print("DATASET:")
print(df)

print("df.head() & df.tail():")
print(df.head())
print(df.tail())

x=df.iloc[:,:-1].values
print("X_VALUES:")
print(x)
y=df.iloc[:,1].values
print("Y_VALUES:")
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=1/3,random_state=0) 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)
print("Y_Prediction Values:")
print(Y_pred)
print("Y_Test Values:")
print(Y_test)

print("TRAINING DATASET GRAPH:")
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("TEST DATASET GRAPH:")
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("VALUES FOR MSE, MAE AND RMSE:")
mse=mean_squared_error(Y_test,Y_pred) 
print('MSE = ',mse) 
mae=mean_absolute_error(Y_test,Y_pred) 
print('MAE = ',mae) 
rmse=np.sqrt(mse) 
print('RMSE = ',rmse)
```



## Output:

![Screenshot 2024-04-02 130631](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/d4174789-86d7-4420-adcc-9f8d764ad80d)
![Screenshot 2024-04-02 130708](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/f2a6109e-aaa9-42a7-87cc-b7dbd9bd5a1f)
![Screenshot 2024-04-02 130718](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/6f91dc5e-135a-49ed-b25a-f58ea208c32f)
![Screenshot 2024-04-02 130737](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/d8ddcf68-0860-45aa-bc64-6ce771d299bf)
![Screenshot 2024-04-02 130754](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/b2b1b40b-b742-45a0-8d0d-1b5dabf1295d)
![Screenshot 2024-04-02 130810](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/0d14899c-e850-4029-ab75-6afa7e081cea)
![Screenshot 2024-04-02 130828](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/2548b20f-3ca1-4699-89ee-3f39da71ade0)
![Screenshot 2024-04-02 130840](https://github.com/sravanipopuri2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139778301/b564f32a-a1d7-4d50-8070-1b747fb04a16)










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
