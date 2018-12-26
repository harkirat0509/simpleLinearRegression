#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Aim: Develop Linear Regression model able to predict CO2 Emissions from Engine Size
"""
Created on Tue Dec 25 14:55:50 2018

@author: harkirat
"""
#importing libraries and dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#reading of dataset
data = pd.read_csv('FuelConsumptionCo2.csv')

#distribution of dataset
np.random.seed(1)
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]

#getting no of data points in both train and test set
m_train = len(train)
m_test = len(test)
weights = np.zeros((2,1))

#initialising numpy array to desired shape
train_data_X = np.zeros((weights.shape[0],m_train))
test_data_X = np.zeros((weights.shape[0],m_test))

#first row will correspond to bias feature
train_data_X[0,:] = np.ones((1,m_train))
test_data_X[0,:] = np.ones((1,m_test))

#corresponding rows will correspond to independent features
train_data_X[1:,:] = train['ENGINESIZE']
test_data_X[1:,:] = test['ENGINESIZE']

#row correspond to dependent variable values
train_data_Y = np.zeros((1,m_train))
train_data_Y[0,:] = train['CO2EMISSIONS']
test_data_Y = np.zeros((1,m_test))
test_data_Y[0,:] = test['CO2EMISSIONS']

#costFunction
def costFunction(data,labels,parameters):
    '''
    Cost Function is used to calculate cost or loss incurred due to model, input parameters are data, correct labels and corresponding parameters
    return cost
    '''
    predicted = np.dot(parameters.T,data)
    cost = np.sum(np.power(predicted-labels,2))/(2*labels.shape[1])
    return cost

def learningModel(data,labels,parameters,learning_rate=0.001,num_iterations=1000):
    '''
    learningModel is used for training the model, Input parameters are data to be trained on, correct labels to check for loss, parameters to be updated for, learning_rate and number of iterations required
    return learned parameters
    '''
    cost = []
    for i in range(num_iterations):
        predicted = np.dot(parameters.T,data)
        testing = np.sum((predicted-labels)*data,axis=1).reshape(len(parameters),1)
        parameters = parameters - (learning_rate/labels.shape[1])*(testing)
        cost.append(costFunction(data,labels,parameters))
        
    plt.plot(range(num_iterations),cost)
    return parameters

print('Testing Learning Rate on Training Data')
#learning rate will vary between 10^-5 to 10^-1
learning = -4 * np.random.rand(11)-1
learning = np.power(10,learning)

#used for collecting r2score and parameters of each testing so to choose afterwards
modelSelection=[]
for i in learning:
    parameters = learningModel(train_data_X,train_data_Y,weights,i)
    train_cost = costFunction(train_data_X,train_data_Y,parameters)
    test_cost = costFunction(test_data_X,test_data_Y,parameters)
    print('For Learning rate',i)
    print('Training Cost',train_cost)
    print('Test Cost',test_cost)
    print('R2 Score')
    print(r2_score(test_data_Y.T,np.dot(parameters.T,test_data_X).T))
    print('')
    modelSelection.append([parameters,r2_score(test_data_Y.T,np.dot(parameters.T,test_data_X).T)])

#model selection on basis of r2_score
max_r2 = 0
for i,model in enumerate(modelSelection):
    if model[1]>max_r2:
        max_r2 = model[1]
        index = i
    
#final Model selection
print('Final Model Selection')
print('Learning rate: ',learning[index])
print('Corresponding R2_score: ',modelSelection[index][1])
print('Corresponding Parameters: ',modelSelection[index][0])




    



















