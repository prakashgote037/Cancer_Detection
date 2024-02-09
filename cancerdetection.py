#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the dependencies

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# In[2]:


# Read the csv file from notebook with help of pandas library
data_frame = pd.read_csv('data.csv')

# Please show the first five rows of dataframe
data_frame.head()

# In[3]:


# Please show the complete data of the dataframe

data_frame

# In[4]:


# We have copy orignal data into another dataframe to do some experiment

data = data_frame.copy()

# In[5]:


# Please show the shape of dataframe i.e rows and columns

data.shape

# In[6]:


# Please show me the info of dataframe

data.info()

# In[7]:


# Data of diagnosis columns is converted into Binary Data

df_one = pd.get_dummies(data['diagnosis'])

# In[8]:


# Please show the first five rows of dataframe

df_one.head()

# In[9]:


# Binary Data is Concatenated into Dataframe

df_two = pd.concat((df_one, data), axis=1)

# In[10]:


# Please show the first five rows of dataframe

df_two.head()

# In[11]:


# Diagnosis column is dropped

df_two = df_two.drop(['diagnosis'], axis=1)

# In[12]:


# We want M =0 and B =1 So we drop B column here

df_two = df_two.drop(['B'], axis=1)

# In[13]:


# Please show the first five rows of dataframe

df_two.head()

# In[14]:


# False = No Cancer & True = Cancer
# We have rename the column M to diagnosis

data2 = df_two.rename(columns={'M': 'diagnosis'})

# In[15]:


# Please show the complete data of the dataframe

data2

# In[16]:


# We have drop columns name 'Unnamed: 32'

data2 = data2.drop(columns='Unnamed: 32', axis=1)

# In[17]:


# We have drop columns name 'id'

data2 = data2.drop(columns='id', axis=1)

# In[18]:


# statistical measures about data

data2.describe()

# In[19]:


# checking the distrubution of target variable

data2['diagnosis'].value_counts()

# In[20]:


# True = Malignant
# False = Benign

# We have perform groupby operations for diagnosis coloumns

data2.groupby('diagnosis').mean()

# In[21]:


# Separating the features and target

X = data2.drop(columns='diagnosis', axis=1)
Y = data2['diagnosis']

# In[22]:


# Splitting data into training data and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# In[23]:


# Please show the shape of dataframe i.e rows and columns

(X.shape, X_train.shape, X_test.shape)

# In[24]:


# Please show the complete data of the dataframe

X_train

# In[25]:


# Model Training

# Here We are creating model using Logistic Regression

model = LogisticRegression()

# In[26]:


# We have given training data to model for training the model

model.fit(X_train, Y_train)

# In[31]:


# Model Evaluation

# Here We are evaluating model with metrics called 'Accuracy Score'

# Accuracy of model on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

Accuracy_of_model_on_training_data = training_data_accuracy

print('Accuracy_of_model_on_training_data = ', Accuracy_of_model_on_training_data)

# In[32]:


# Accuracy of the model on testing data

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)

Accuracy_of_the_model_on_testing_data = testing_data_accuracy

print('Accuracy_of_the_model_on_testing_data = ', Accuracy_of_the_model_on_testing_data)

# In[33]:


# Building Predictive System

input_data = (
9.504, 12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076, 0.1815, 0.06905, 0.2773, 0.9768, 1.909, 15.7, 0.009606,
0.01432, 0.01985, 0.01421, 0.02027, 0.002968, 10.23, 15.66, 65.13, 314.9, 0.1324, 0.1148, 0.08867, 0.06227, 0.245,
0.07773)

# change input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# We have put data into predict function for prediction
prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[True] == True):
    print('The Breast Cancer is Malignant')
else:
    print('The Breast Cancer is Benign')

# In[ ]:
