# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 23:50:41 2022

@author: Galal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
path = "C:\\Users\\Galal\\Downloads\\My-Github\\CV_learning\\Sprint 3\\Sprint 3\\Andrew\\ex1data1.txt"
data = pd.read_csv(path,header=None,names=['Population','Profit'])

#show data details
#print("data = \n",data.head(10))
#print(data.describe())
data.plot(kind='scatter',x='Population',y='Profit',figsize=(5,5))

data.insert(0,'Ones',1)
#print(data.head(10))

#Seperate x(training) from y (target)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

#convert from data fram to numpy matrix
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# cost function
def computeCost(x,y,theta):
    z = np.power(((X*theta.T)-y),2)
    return np.sum(z)/(2*len(X))

print(computeCost(X,y,theta))

#Gradient Descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , computeCost(X, y, g))
print('**************************************')
#=========================================================================

# get best fit line

x = np.linspace(data.Population.min(), data.Population.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)




# draw the line

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
