# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial values for the weights (w) and bias (b).
2. Compute Predictions: Calculate the predicted probabilities using the logistic function.
3. Compute Gradient: Compute the gradient of the loss function with respect to w and b.
4. Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat steps 2-4 until convergence or a maximum number of iterations is reached.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MANOGARAN S
RegisterNumber:  212223240081
*/


import pandas as pd
data=pd.read_csv("Employee (1).csv")
data.head()

data.info()

data.isnull()

data.isnull().sum()

data['left'].value_counts()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

y=data['left']
y.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![Screenshot 2024-03-28 074824](https://github.com/manogarans/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331782/dc4a5693-5a8c-4535-8abf-b086897e7eeb)
![Screenshot 2024-04-05 205134](https://github.com/manogarans/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331782/05b6727b-2e5b-415d-b45c-b60d1377a6b1)
![Screenshot 2024-04-05 205211](https://github.com/manogarans/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331782/232a877b-0d0d-4dae-ae6d-940657ed0a90)
![Screenshot 2024-04-05 205306](https://github.com/manogarans/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331782/fca3ed3d-2af1-4227-9880-c204b7bb7c69)
![Screenshot 2024-04-05 205330](https://github.com/manogarans/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331782/797e7ec6-83fd-485a-87b3-7a8a12a3bb1d)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

