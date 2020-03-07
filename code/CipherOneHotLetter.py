#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine learning Ciphertext with Plain Word and Cipher Letter
Created on Sun Feb 23 22:26:33 2020

@author: C Kornafel
This version of the dataset is separated by unique word in the plaintext and the 
word vectors include the corresponding cipher letters which will be onehot encoded 
for the machine learning predictions. 

This approach allows for a reduced amount of classifiers with multiple cipher vectors
that can be used in the various predictions.

The dataset used will be the "matched" plaintext and ciphertext words from the level 1
category. However, if matching is unknown, a similar dataset could be created by matching
word frequencies and following the same measurements scheme.   
"""
import pandas as pd
import numpy as np
import random

#Dataset Prep
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Decision Tree
from sklearn.tree import DecisionTreeClassifier 

##Support Vector Machine
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

#KNN
from sklearn.neighbors import KNeighborsClassifier 

#H2O Neural Network
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch


ltr = pd.read_csv('/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/cipher_ltr.csv')
ltr = ltr.fillna(0) #Filling in the missing values with a 0

ltr.head()

# ltr.dtypes

#Making a sample size dataframe for complex machine learning
random.seed()
SAMPLE_SIZE = 1000
ltr_samp = ltr.sample(n=SAMPLE_SIZE)

plain_target = ltr['plainwd'].astype('category')
p_target_samp = ltr_samp['plainwd'].astype('category')

ltr_f = ltr.drop('plainwd', axis = 1)
ltr_samp = ltr_samp.drop('plainwd', axis = 1)

one_hot = pd.get_dummies(ltr_f)
one_hot_samp = pd.get_dummies(ltr_samp)

#Checking data before proceeding
one_hot.head()
plain_target.head()

one_hot_samp.head()
p_target_samp.head()


#Establishing the features and target values
features = one_hot
target = plain_target

features_samp = one_hot_samp
target_samp = p_target_samp


#Next, dividing data into test and train sets
x_train, x_test, y_train, y_test = train_test_split(features, target, random_state = 0)

#Scaling the datasets
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#Sample Size Data
x_samp_train, x_samp_test, y_samp_train, y_samp_test = train_test_split(features_samp, target_samp, random_state = 0)

#Scaling the datasets
scaler1 = StandardScaler()
scaler1.fit(x_samp_train)

x_samp_train = scaler1.transform(x_samp_train)
x_samp_test = scaler1.transform(x_samp_test)

#Encoding the target variable/classes
enc = LabelEncoder()
y_test = enc.fit_transform(y_test)
y_train = enc.fit_transform(y_train)

y_samp_test = enc.fit_transform(y_samp_test)
y_samp_train = enc.fit_transform(y_samp_train)

############   Decision Tree   ##########################

#Global Varlaibles 
MAX_DEPTH = 100

#Creating the model
tree_mod = DecisionTreeClassifier(max_depth = MAX_DEPTH)
tree_mod.fit(x_train, y_train)

tree_pred = tree_mod.predict(x_test)

#Confusion matrix of results
tree_cm = confusion_matrix(y_test, tree_pred)
print(tree_cm)

tree_err = np.mean(tree_pred != y_test)

##Reducing the dataset size so I can increase depth
#Creating the model without max depth value - this will fully extend each leaf
s_tree_mod = DecisionTreeClassifier()
s_tree_mod.fit(x_samp_train, y_samp_train)

s_tree_pred = s_tree_mod.predict(x_samp_test)

#Confusion matrix of results
s_tree_cm = confusion_matrix(y_samp_test, s_tree_pred)
print(s_tree_cm)

s_tree_err = np.mean(s_tree_pred != y_samp_test)

s_tree_cr = classification_report(y_samp_test, s_tree_pred, output_dict=True)
s_tree_report = pd.DataFrame(s_tree_cr).transpose()
s_tree_report.to_csv('tree_report',index = False)

print( "Accuracy of Decision Tree: ", s_tree_cr["accuracy"])
print( "Precision of Decision Tree: ", s_tree_cr["macro avg"]["precision"])
print( "F1-Score of Decision Tree: ", s_tree_cr["macro avg"]["f1-score"])


############   KNN    ########################## 

#Global Variables
NEIGHBORS = 10 

#Creating and fitting the KNN model
knn_mod = KNeighborsClassifier(n_neighbors = NEIGHBORS)
knn_mod.fit(x_train, y_train)

#Determining Accuracy
knn_mod_accuracy = knn_mod.score(x_test,y_test)

#Predict
knn_pred = knn_mod.predict(x_test)

#Confusion matrix of results
knn_cm = confusion_matrix(y_test, knn_pred)

knn_cr = classification_report(y_test, knn_pred, output_dict=True)
knn_report = pd.DataFrame(knn_cr).transpose()
knn_report.to_csv('knn_report',index = False)

##Reducing the dataset size so I can perform some fine tuning and hopefully improve results

#Finding the K with the lowest error
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_samp_train, y_samp_train)
    pred_i = knn.predict(x_samp_test)
    error.append(np.mean(pred_i != y_samp_test))

#All 50 itterations had an error of 100%

#Atempting to find optimal k with original size set
error = []
for i in range(10,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

#This ran for 1.5 hours and did not produce a result

from collections import Counter
Counter(error).keys()

NEIGHBORS = 100
s_knn_mod = KNeighborsClassifier(n_neighbors = NEIGHBORS)
s_knn_mod.fit(x_samp_train, y_samp_train)

#Determining Accuracy
s_knn_mod_accuracy = s_knn_mod.score(x_samp_test,y_samp_test)
print("KNN Accuracy: ", s_knn_mod_accuracy)

#Predict
s_knn_pred = s_knn_mod.predict(x_samp_test)

#Confusion matrix of results
knn_cm = confusion_matrix(y_test, knn_pred)

knn_cr = classification_report(y_samp_test, s_knn_pred, output_dict=True)
knn_report = pd.DataFrame(knn_cr).transpose()
knn_report.to_csv('knn_report',index = False)

print( "Accuracy of KNN: ", knn_cr["accuracy"])
print( "Precision of KNN: ", knn_cr["macro avg"]["precision"])
print( "F1-Score of KNN: ", knn_cr["macro avg"]["f1-score"])

# ############   SVM    ########################## 

#Global Variables
C = 5
from sklearn.svm import LinearSVC
#Creating and fitting the SVM models
svm_mod_linear = LinearSVC()
svm_mod_linear.fit(x_samp_train, y_samp_train)

svm_mod_ovo = SVC(decision_function_shape = 'ovo')
svm_mod_ovo.fit(x_samp_train, y_samp_train)

#Determing Accuracy
svm_lin_accuracy = svm_mod_linear.score(x_samp_test, y_samp_test)
svm_ovo_accuracy = svm_mod_ovo.score(x_samp_test, y_samp_test)


#Tuning the model
svc= SVC()
param = {'kernel':('linear', 'rbf', 'poly'), 'C': [1,2,3,4,5,6,7,8,9,10]}
grid = GridSearchCV(svc,param, cv=2)


grid_info = grid.fit(x_samp_train, y_samp_train)

tune_info = grid_info.best_params_

#Rerunning with updated parameters
C = 1
svm_mod_linear = SVC(kernel = 'linear', C = C)
svm_mod_linear.fit(x_samp_train, y_samp_train)


#Predict
svm_ovo_pred = svm_mod_ovo.predict(x_samp_test)


#Confusion Matrix
svm_ovo_cm = confusion_matrix(y_samp_test, svm_ovo_pred)


svm_ovo_cr = classification_report(y_samp_test, svm_ovo_pred, output_dict=True)
print( "Accuracy of SVM OvO: ", svm_ovo_cr["accuracy"])
print( "Precision of SVM OvO: ", svm_ovo_cr["macro avg"]["precision"])
print( "F1-Score of SVM OvO: ", svm_ovo_cr["macro avg"]["f1-score"])

############  H2O  NeuralNetwork    ########################## 

h2o.init() #Initializing the neural network

#Loading the modified files into the network
c_hex = h2o.import_file('/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/c_ltr_h2o.csv')

response = 'plainwd'

temp = c_hex.col_names
temp.remove('plainwd')
predictors = temp

#Splitting the H2O frame into train 80% test 20% 
train, valid = c_hex.split_frame(ratios=[0.8], seed = 1234)

#According to the H2O documentation, categorical data is automatically one-hot encoded


#Create the h2o model
model = H2ODeepLearningEstimator(
    model_id = 'ltr_model1',
    distribution = "multinomial",
    activation = "RectifierWithDropout",
    hidden = [32,32,32],
    input_dropout_ratio = 0.2,      
    sparse = True,
    epochs = 10, 
    variable_importances = True)

model.train(
        x=predictors,
        y=response,
        training_frame=train,
        validation_frame=valid)


#Obtaining the Performance of the Model
model.mse(valid=True)
#The MSE value of over 0.998 is quite large and indicates that the model does not fit well. 
#This was very similar to the MSE error achived with the initial dataset


#Tuning the model using a 3-fold cross validation
model_cf = H2ODeepLearningEstimator(
    model_id = 'ltr_model1',
    distribution = "multinomial",
    activation = "RectifierWithDropout",
    hidden = [32,32,32],
    input_dropout_ratio = 0.2,      
    sparse = True,
    epochs = 10,
    variable_importances = True, 
    nfolds = 3)

model_cf.train(
    x = predictors,
    y = response,
    training_frame = train)

#Checking the error on the cross validated model
model_cf.mse(xval = True)
#This yielded the same high error rate 

#Grid Search to tune parameters
hidden_param = [[40,32,40], [40,40,40], [50,50,50]]
ll_param = [1e-2, 1e-3, 1e-5]
params = {"hidden" : hidden_param, "l1":ll_param}

model_grid = H2OGridSearch(H2ODeepLearningEstimator, 
                           hyper_params = params) 

model_grid.train(x=predictors, 
                 y = response,
                 distribution = 'multinomial', 
                 epochs = 50,
                 score_interval = 2,
                 stopping_rounds = 2, 
                 stopping_tolerance = 0.05,
                 stopping_metric = "misclassification",
                 training_frame = train,
                 validation_frame=valid)

for model in model_grid:
    print(model.model_id + " mse: " + str(model.mse()))

model_up = H2ODeepLearningEstimator(
    model_id = 'ltr_model1',
    distribution = "multinomial",
    activation = "RectifierWithDropout",
    hidden = [100,100],
    input_dropout_ratio = 0.2,      
    sparse = True,
    l1 = 0.01,
    epochs = 39, 
    variable_importances = True)

model_up.train(
        x=predictors,
        y=response,
        training_frame=train,
        validation_frame=valid)

model_up.accuracy()



h2o.cluster().shutdown()






