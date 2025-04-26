# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas, numpy, and matplotlib libraries.  
2. Load the Placement Data CSV and display its info.  
3. Drop 'sl_no' and 'salary' columns from the dataset.  
4. Convert selected columns to categorical type and encode them numerically.  
5. Separate features (`x`) and target (`y`), initialize random theta.  
6. Define sigmoid, loss function, and gradient descent for training.  
7. Train model using gradient descent and update theta.  
8. Define predict function and make predictions on training data.  
9. Calculate and print training accuracy.  
10. Predict outcomes for two new input samples and print results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: V RAKSHITA
RegisterNumber:  212224100049
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("Placement_Data.csv")
data.info()
data=data.drop(['sl_no','salary'],axis=1)
data
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
x=data.iloc[:,:-1].values
y=data.iloc[:,-1]
theta = np.random.randn(x.shape[1])
    
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta  
    
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
    
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
    
y_pred = predict(theta, x)
    
accuracy=np.mean(y_pred.flatten()==y)
    
print("Acuracy:",accuracy)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,5,65,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

![Screenshot (277)](https://github.com/user-attachments/assets/b0f889e8-7fda-42db-b01a-6e79a36a3815)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

