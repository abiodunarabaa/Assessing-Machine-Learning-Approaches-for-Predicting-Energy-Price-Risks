#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, linear_model, metrics
from sklearn.ensemble import RandomForestRegressor


# # Dataset Reading and Exploration

# <B>Reading the dataset</B>

# In[129]:


df = pd.read_csv("CE802_P3_Data.csv")


# <B>Checking for null value in dataset</B>

# In[130]:


display(df.isnull().sum())


# <b> Skewness of the Dataset

# In[131]:


df.skew()


# <b> Columns Data types <b/>

# In[132]:


print(df.dtypes)
data_full_train.head()


# <b> Correlation of the Features <b/>

# In[133]:


corr = df.corr()
plt.figure(figsize = (20,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, center=0, cmap="YlGnBu", annot=True)
plt.show()


# 

# 

# # Comparative Analysis

# <b> Reading the dataset<b/>

# In[135]:


df = pd.read_csv("CE802_P3_Data.csv")


# <b> Dropping attributes F6 and F9 being an object datatype<b/>

# In[136]:


df_dropF6andF9 = df
df_dropF6andF9.drop(columns=['F6', 'F9'], inplace = True)


# <b> Separating the feactures and target <b/>

# In[137]:


df_target = df_dropF6andF9["Target"]
df_feat = df_dropF6andF9.drop(columns = ["Target"],axis = 1)


# <b> Normalization of the features for K-NN and SVM <b/>

# In[138]:


scaler = StandardScaler()
scaler.fit(df_feat)
df_feat = scaler.transform(df_feat)


# <b> Data splitting to verify model accuracy after cross validation and gridsearch <b/>

# In[139]:


data_feat_train, data_feat_test, data_target_train, data_target_test = train_test_split(df_feat, df_target, test_size=0.25)


# # Method 1: Linear Regression

# In[140]:


regr=linear_model.LinearRegression() #Linear regression object
regr.fit(data_feat_train,data_target_train) #Use training sets to train the mode


# In[141]:


data_target_test_pred=regr.predict(data_feat_test) #Make predictions
data_target_train_pred=regr.predict(data_feat_train)


# In[142]:


# regression coefficients
#print('Coefficients: ', regr.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(regr.score(data_feat_test,data_target_test)))
  
# plot for residual error
  
## setting plot style
plt.style.use('fivethirtyeight')
  
## plotting residual errors in training data
plt.scatter(regr.predict(data_feat_train), regr.predict(data_feat_train) - data_target_train, color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(regr.predict(data_feat_test), regr.predict(data_feat_test) - data_target_test, color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")
  
## method call for showing the plot
plt.show()


# In[143]:


mean_squared_error(data_target_test,data_target_test_pred)


print('MSE train: %.3f, test: %.3f' % (mean_squared_error(data_target_train, data_target_train_pred),
                                       mean_squared_error(data_target_test,data_target_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(data_target_train, data_target_train_pred),
                                       r2_score(data_target_test,data_target_test_pred)))  #Variance score


# In[144]:


plt.scatter(data_target_test,data_target_test_pred,color ='lavender')


# In[ ]:





# # Method 2: Decision Tree Regressor

# In[145]:


decisiontree_regressor = DecisionTreeRegressor(max_depth=10, max_features=34, random_state = 2)
decisiontree_regressor.fit(data_feat_train,data_target_train)
data_target_test_pred = decisiontree_regressor.predict(data_feat_test)
data_target_train_pred = decisiontree_regressor.predict(data_feat_train)
print(decisiontree_regressor)


# In[146]:


print('MSE train: %.3f, test: %.3f' % (mean_squared_error(data_target_train, data_target_train_pred),
                                       mean_squared_error(data_target_test, data_target_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(data_target_train, data_target_train_pred),
                                       r2_score(data_target_test, data_target_test_pred)))


# In[147]:


plt.figure(figsize=(10, 8))
plt.scatter(data_target_train_pred, data_target_train_pred - data_target_train, c='grey', marker='o', s=35, alpha=0.65, edgecolor='k', label='Training data')
plt.scatter(data_target_test_pred, data_target_test_pred - data_target_test, c='lightgreen', marker='s',s=35, alpha=0.7, edgecolor='k', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()


# In[ ]:





# # Method 3: Random Forest Regressor

# In[148]:


forest = RandomForestRegressor(n_estimators=450, criterion='squared_error', max_depth=16, random_state=5, n_jobs=2)
forest.fit(data_feat_train, data_target_train)
y_train_pred = forest.predict(data_feat_train)
y_test_pred = forest.predict(data_feat_test)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(data_target_train, y_train_pred),
                                       mean_squared_error(data_target_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(data_target_train, y_train_pred),
                                       r2_score(data_target_test, y_test_pred)))


# In[149]:


plt.figure(figsize=(10, 8))
plt.scatter(y_train_pred, y_train_pred - data_target_train, c='grey', marker='o', s=35, alpha=0.65, edgecolor='k', label='Training data')
plt.scatter(y_test_pred, y_test_pred - data_target_test, c='lightgreen', marker='s',s=35, alpha=0.7, edgecolor='k', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()


# # 

# # Prediction on a hold-out test dataset using Forest Tree Classifier

# In[150]:


#Reading the dataset for training the model and dropping columns: F6 and F9

data_training = pd.read_csv("CE802_P3_Data.csv")
data_target_train = data_training["Target"]
data_full_train = data_training.drop(columns = ['F6', 'F9','Target'],axis = 1)
forest = RandomForestRegressor(n_estimators=450, criterion='squared_error',max_depth=16, random_state=5, n_jobs=2)

forest.fit(data_full_train, data_target_train) #training the dataset


# In[151]:


#Readng the Test dataset
data_pred= pd.read_csv("CE802_P3_Test.csv")
data_full_pred = data_pred.drop(columns = ['F6', 'F9','Target'],axis = 1)

test_pred = forest.predict(data_full_pred) #predicting the target column


# In[152]:


pred =  pd.DataFrame(test_pred,columns=['Target'])  #loading the predicted values to the Test file
final_pred = pd.read_csv('CE802_P3_Test.csv')
final_pred.drop(columns = ['Target'],inplace = True)
submit_csv = pd.concat([final_pred,pred],axis=1)
submit_csv.to_csv("CE802_P3_Test.csv", index = False)
print("Completed: Target values populated")


# In[ ]:





# In[ ]:




