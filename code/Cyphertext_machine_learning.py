#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:58:43 2020

@author: C Kornafel
Attempt at ciphertext decryption using only plaintext for training examples. 
Using H2O Neural Network and SVM supervised learning
"""
import pandas as pd


train = pd.read_csv('pred_train.csv')
test1 = pd.read_csv('pred_test1.csv')

train.head(10)
test1.head(10)

train_sample = train.sample(n=900, random_state=1)
test1_sample = test1[test1['index'].isin(train_sample['index'])]

train_sample.head(10)
test1_sample.head(10)

#Vlaidating that all test classes are still in train
test1_sample['index'].isin(train['index']).value_counts()

#Changing the response variable to a string
train['index'] = train['index'].apply(str)
test1['index'] = test1['index'].apply(str)

train.dtypes
test1.dtypes


#Using the h2o neural network to predict ciphertext
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


h2o.init() #Initializing the neural network

#Loading the modified files into the network
train_hex = h2o.H2OFrame(train)
test1_hex = h2o.H2OFrame(test1)


##Updating with the sample size of the ciphertext
sTrain_hex = h2o.H2OFrame(train_sample)
sTest1_hex = h2o.H2OFrame(test1_sample)

#Changing the dep variable to a factor
train_hex['index'] = train_hex['index'].asfactor()
test1_hex['index'] = test1_hex['index'].asfactor()


sTrain_hex['index'] = sTrain_hex['index'].asfactor()
sTest1_hex['index'] = sTest1_hex['index'].asfactor()

#identifying the response and predictor variables
r = "index"
p = train_hex.names[3:60]

#Create the h2o model
model = H2ODeepLearningEstimator(
        distribution = "multinomial",
        activation = "RectifierWithDropout",
        hidden = [32,32,32],
        input_dropout_ratio = 0.2,      
        sparse = True,
        epochs = 10, 
        variable_importances = True)

model.train(
        x=p,
        y=r,
        training_frame=train_hex,
        validation_frame=test1_hex)

#Training the Model with Sample Dataset
model.train(
        x=p,
        y=r,
        training_frame=sTrain_hex,
        validation_frame=sTest1_hex)

#Obtaining the Performance of the Model
model.mse(valid=True)

#The MSE value of over 0.998 is quite large and indicates that the model does not fit well. 

#Assessing the variable importances for this model 
model.varimp()





##Trying a crossfold model to see if fit can be improved
model_cf = H2ODeepLearningEstimator(
        distribution = "auto",
        activation = "RectifierWithDropout",
        hidden = [32,32,32],
        input_dropout_ratio = 0.2,  
        l1=1e-5,
        sparse = True,
        epochs = 10,
        nfolds=5, 
        variable_importances=True)

#Training the cross-fold model with samole datasets
model_cf.train(
        x=p,
        y=r,
        training_frame=sTrain_hex)


#Obtaining Fit MSE
model_cf.mse(xval=True)

#The cross fold model actually fit worse than the "trial" model as the
#MSE value increased to 0.998!

#Adding more hidden layers and decreasing the epochs to see if it could affect performance

model2_cf = H2ODeepLearningEstimator(
        distribution = "auto",
        activation = "RectifierWithDropout",
        hidden = [50,32,100],
        input_dropout_ratio = 0.2,  
        sparse = True,
        epochs = 3,
        nfolds=5)

model2_cf.train(
        x=p,
        y=r,
        training_frame=sTrain_hex)

#Obtaining Fit MSE
model2_cf.mse(xval=True)

#The MSE keeps getting larger. 

model3_cf = H2ODeepLearningEstimator(
        distribution = "auto",
        activation = "RectifierWithDropout",
        hidden = [50,100,50],
        input_dropout_ratio = 0.02,  
        sparse = True,
        epochs = 3,
        nfolds=5)

model3_cf.train(
        x=p,
        y=r,
        training_frame=sTrain_hex)

model3_cf.mse(xval=True)

#Using initial model to attempt to predict the ciphertext
pred = model3_cf.predict(sTest1_hex)
pred.head()

h2o.cluster().shutdown()





##Support Vector Machine
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Performing a little clean-up for SVM
#removing dependent variable and rowname column
#also, SVMs do not like text, so I am removing the text column as well
sv_train = train
sv_train = sv_train.drop(['index','Unnamed: 0', 'text'], axis = 1)

sv_test = test1
sv_test = sv_test.drop(['index','Unnamed: 0', 'text'], axis = 1)


#Next, splitting off the dependent variable
train_lbl = train.index.values.astype(object)
test_lbl = test1.index.values.astype(object)


#Encoding Response variable
encoder = preprocessing.LabelEncoder()
encoder.fit(train_lbl)
y_train = encoder.transform(train_lbl)

encoder.fit(test_lbl)
y_test = encoder.transform(test_lbl)


#Scaling the values in the datasets
std_scaler = StandardScaler()
sv_train = std_scaler.fit_transform(sv_train)
sv_test = std_scaler.fit_transform(sv_test)


#Building a grid search model 
#grid_mdl = [{'kernel' :['rbf'], 'gamma':[1e-2, 1e-5], 'C' : [1,5,10,100]},
#             {'kernel':['linear'], 'C' : [1,5,10,100]}]

#Tuning parameters for optim SVM fit
sv_mdl = OneVsRestClassifier(SVC(gamma = 'auto'))

sv_mdl.fit(sv_train, y_train)

##The model never completed fitting by using plain text only

#Cleaning Dataset - Attempting with sample size
sv_trainSmp = train_sample
sv_trainSmp = sv_trainSmp.drop(['index','Unnamed: 0', 'text'], axis = 1)

sv_testSmp = test1_sample
sv_testSmp = sv_testSmp.drop(['index','Unnamed: 0', 'text'], axis = 1)

#Creating Response - Sample dataset
trainSmp_lbl = train_sample.index.values.astype(object)
testSmp_lbl = test1_sample.index.values.astype(object)

#Encoding Response - Sample Datasets
encoder.fit(trainSmp_lbl)
y_trainSmp = encoder.transform(trainSmp_lbl)

encoder.fit(testSmp_lbl)
y_testSmp = encoder.transform(testSmp_lbl)

#Scaling - Sample Size
sv_trainSmp = std_scaler.fit_transform(sv_trainSmp)
sv_testSmp = std_scaler.fit_transform(sv_testSmp)


#Attempting with Sample Datasets
sv_mdlSmp = OneVsRestClassifier(SVC(gamma = 'auto'))

sv_mdlSmp.fit(sv_trainSmp, y_trainSmp)

predSmp = sv_mdlSmp.predict(sv_testSmp)

print("Accuracy: ", metrics.accuracy_score(y_testSmp, predSmp))


#The sample dataset of plaintext-only did not achieve any accuracy



