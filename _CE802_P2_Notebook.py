#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# # Dataset Reading and Exploration

# <B>Reading the dataset</B>

# In[117]:


df = pd.read_csv("CE802_P2_Data.csv")


# <B>Checking for null value in dataset</B>

# In[112]:


display(df.isnull().sum())


# <b> Skewness of the Dataset

# In[113]:


df.skew()


# <b> Columns Data types <b/>

# In[114]:


df.dtypes


# <b> Correlation of the Features <b/>

# In[5]:


corr = df.corr()
plt.figure(figsize = (15,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, center=0, cmap="YlGnBu", annot=True)
plt.show()


# 

# 

# # Method 1: Dropping Column F21

# <b> Reading the dataset<b/>

# In[78]:


df = pd.read_csv("CE802_P2_Data.csv")


# <b> Attribute F21 has 500 Null cells, hence dropping it <b/>

# In[79]:


df_dropf21 = df
df_dropf21.drop(columns=['F21'], inplace = True)


# <b> Separating the feactures and target <b/>

# In[80]:


df_dropf21_class = df_dropf21["Class"]
df_dropf21_feat = df_dropf21.drop(columns = ["Class"],axis = 1)


# <b> Normalization of the features for K-NN and SVM <b/>

# In[81]:


scaler = StandardScaler()
scaler.fit(df_dropf21_feat)
df_dropf21_feat = scaler.transform(df_dropf21_feat)


# <b> Data splitting to verify model accuracy after cross validation and gridsearch <b/>

# In[82]:


data_feat_train, data_feat_test, data_class_train, data_class_test = train_test_split(df_dropf21_feat,df_dropf21_class,test_size=0.25,stratify=df_dropf21_class,random_state=1234)


# <b> Pruned Decision Tree <b/>

# In[83]:


clf_tree = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234) #Decision tree with grid search

param_grid = {'max_depth': np.arange(4,21),'min_samples_split': np.arange(4,21),'min_samples_leaf': np.arange(4,21),
              'max_features': ['sqrt','auto','log2']}

tree_gridcv = GridSearchCV(clf_tree,param_grid,cv=10 ,n_jobs=-1)
tree_gridcv.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(tree_gridcv.best_params_))
print("The best score: " + str(tree_gridcv.best_score_))


# <b> Cross Validation of Pruned Decision Tree <b/>

# In[84]:


clf_tree_prunned = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234,
                                               max_depth= tree_gridcv.best_params_['max_depth'],
                                               min_samples_leaf= tree_gridcv.best_params_['min_samples_leaf'],
                                               min_samples_split=tree_gridcv.best_params_['min_samples_split'] )

score_tree = cross_val_score(clf_tree_prunned,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation the whole data set, using the best parameters by gridsearch
print('The average accuracy:', np.mean(score_tree))


clf_tree_prunned.fit(data_feat_train,data_class_train) #compute the confussion matrix by splitting the data into trainning and testing
tree_pred = clf_tree_prunned.predict(data_feat_test)
print(confusion_matrix(data_class_test, tree_pred))
print(classification_report(data_class_test, tree_pred))


# <b> K-Nearest Neighbor (K-NN) <b/>

# In[13]:


knn_gridcv = KNeighborsClassifier()
param_gridsearch = {'n_neighbors': np.arange(1,80),'weights':['uniform','distance']} #dictionary with the number of neighbors to try

knn_gridsearch = GridSearchCV(knn_gridcv,param_gridsearch,cv=10)
knn_gridsearch.fit(data_feat_train,data_class_train)
print("The best parameters: " + str(knn_gridsearch.best_params_))
print("The best score: "+ str(knn_gridsearch.best_score_))


# <B> Cross Validation of K-Nearest Neighbor (K-NN) <b/>

# In[14]:


knn_model = KNeighborsClassifier(n_neighbors = knn_gridsearch.best_params_['n_neighbors'],
                                 weights=knn_gridsearch.best_params_['weights'])

score_knn = cross_val_score(knn_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_knn))

knn_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
knn_pred = knn_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, knn_pred))
print(classification_report(data_class_test, knn_pred))


# <b> Support Vector Machine <b/>

# In[15]:


clf_svm = svm.SVC()
param_grid = {'C': np.logspace(-1, 3, 9),  
              'gamma': np.logspace(-7, -0, 8)}

svm_gridsearch = GridSearchCV(clf_svm,param_grid,n_jobs=-1, cv = 10)
svm_gridsearch.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(svm_gridsearch.best_params_))
print("The best score : " + str(svm_gridsearch.best_score_))


# <b> Cross Validation of Support Vector Machine <b/>

# In[16]:


svm_model = svm.SVC(C = svm_gridsearch.best_params_['C'],gamma=svm_gridsearch.best_params_['gamma'])

score_svm = cross_val_score(svm_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_svm))

svm_model.fit(data_feat_train,data_class_train)#confussion matrix by splitting the data into trainning and testing
svm_pred = svm_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, svm_pred))
print(classification_report(data_class_test, svm_pred))


# <b> Random Forest <b/>

# In[17]:


rf = RandomForestClassifier(criterion='entropy',random_state=1234)
param_grid = {'n_estimators':[400,450,500,550,600],'max_depth': np.arange(4,20)}
#'max_depth': np.arange(4,19),'min_samples_split': np.arange(4,19),'min_samples_leaf': np.arange(4,25)}

rf = GridSearchCV(rf, param_grid,cv=10,n_jobs=-1)
rf.fit(data_feat_train,data_class_train)

print("The best parameters: "+ str(rf.best_params_))
print("The best score: " + str(rf.best_score_))


# <b> Cross Validation of Random Forest <b/>

# In[18]:


rf_model = rf = RandomForestClassifier(criterion='entropy',n_estimators= rf.best_params_['n_estimators'],
                                      max_depth=rf.best_params_['max_depth'],random_state=1234)


score_rf = cross_val_score(rf_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_rf))
print(score_rf.std())

rf_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
rf_pred = rf_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, rf_pred))
print(classification_report(data_class_test, rf_pred))


#    

# 

# # Method 2: Replacing F21 with the Mean value

# <b> Reading the dataset<b/>

# In[89]:


df = pd.read_csv("CE802_P2_Data.csv")


# <b> Replacing F21 missing values with mean value of the same column <b/>

# In[103]:


df.fillna(df['F21'].mean(),inplace = True)


# <b> Separating the feactures and target <b/>

# In[91]:


df_f21mean_class = df["Class"]
df_f21mean_feat = df.drop(columns = ["Class"],axis = 1)


# <b> Normalization of the features for K-NN and SVM <b/>

# In[92]:


scaler = StandardScaler()
scaler.fit(df_f21mean_feat)
df_f21mean_feat = scaler.transform(df_f21mean_feat)


# <b> Data splitting to verify model accuracy after cross validation and gridsearch <b/>

# In[93]:


data_feat_train, data_feat_test, data_class_train, data_class_test = train_test_split(df_f21mean_feat,df_f21mean_class,test_size=0.25,stratify=df_f21mean_class,random_state=1234)


# <b> Pruned Decision Tree <b/>

# In[94]:


clf_tree = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234) #Decision tree with grid search

param_grid = {'max_depth': np.arange(4,21),'min_samples_split': np.arange(4,21),'min_samples_leaf': np.arange(4,21),
              'max_features': ['sqrt','auto','log2']}

tree_gridcv = GridSearchCV(clf_tree,param_grid,cv=10 ,n_jobs=-1)
tree_gridcv.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(tree_gridcv.best_params_))
print("The best score: " + str(tree_gridcv.best_score_))


# <b> Cross Validation of Pruned Decision Tree <b/>

# In[95]:


clf_tree_prunned = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234,
                                               max_depth= tree_gridcv.best_params_['max_depth'],
                                               min_samples_leaf= tree_gridcv.best_params_['min_samples_leaf'],
                                               min_samples_split=tree_gridcv.best_params_['min_samples_split'] )

score_tree = cross_val_score(clf_tree_prunned,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation the whole data set, using the best parameters by gridsearch
print('The average accuracy:', np.mean(score_tree))


clf_tree_prunned.fit(data_feat_train,data_class_train) #compute the confussion matrix by splitting the data into trainning and testing
tree_pred = clf_tree_prunned.predict(data_feat_test)
print(confusion_matrix(data_class_test, tree_pred))
print(classification_report(data_class_test, tree_pred))


# <b> K-Nearest Neighbor (K-NN) <b/>

# In[26]:


knn_gridcv = KNeighborsClassifier()
param_gridsearch = {'n_neighbors': np.arange(1,80),'weights':['uniform','distance']} #dictionary with the number of neighbors to try

knn_gridsearch = GridSearchCV(knn_gridcv,param_gridsearch,cv=10)
knn_gridsearch.fit(data_feat_train,data_class_train)
print("The best parameters: " + str(knn_gridsearch.best_params_))
print("The best score: "+ str(knn_gridsearch.best_score_))


# <B> Cross Validation of K-Nearest Neighbor (K-NN) <b/>

# In[27]:


knn_model = KNeighborsClassifier(n_neighbors = knn_gridsearch.best_params_['n_neighbors'],
                                 weights=knn_gridsearch.best_params_['weights'])

score_knn = cross_val_score(knn_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_knn))

knn_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
knn_pred = knn_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, knn_pred))
print(classification_report(data_class_test, knn_pred))


# <b> Support Vector Machine <b/>

# In[28]:


clf_svm = svm.SVC()
param_grid = {'C': np.logspace(-1, 3, 9),  
              'gamma': np.logspace(-7, -0, 8)}

svm_gridsearch = GridSearchCV(clf_svm,param_grid,n_jobs=-1, cv = 10)
svm_gridsearch.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(svm_gridsearch.best_params_))
print("The best score : " + str(svm_gridsearch.best_score_))


# <b> Cross Validation of Support Vector Machine <b/>

# In[29]:


svm_model = svm.SVC(C = svm_gridsearch.best_params_['C'],gamma=svm_gridsearch.best_params_['gamma'])

score_svm = cross_val_score(svm_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_svm))

svm_model.fit(data_feat_train,data_class_train)#confussion matrix by splitting the data into trainning and testing
svm_pred = svm_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, svm_pred))
print(classification_report(data_class_test, svm_pred))


# <b> Random Forest <b/>

# In[30]:


rf = RandomForestClassifier(criterion='entropy',random_state=1234)
param_grid = {'n_estimators':[400,450,500,550,600],'max_depth': np.arange(4,20)}
#'max_depth': np.arange(4,19),'min_samples_split': np.arange(4,19),'min_samples_leaf': np.arange(4,25)}

rf = GridSearchCV(rf, param_grid,cv=10,n_jobs=-1)
rf.fit(data_feat_train,data_class_train)

print("The best parameters: "+ str(rf.best_params_))
print("The best score: " + str(rf.best_score_))


# <b> Cross Validation of Random Forest <b/>

# In[31]:


rf_model = rf = RandomForestClassifier(criterion='entropy',n_estimators= rf.best_params_['n_estimators'],
                                      max_depth=rf.best_params_['max_depth'],random_state=1234)


score_rf = cross_val_score(rf_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_rf))
print(score_rf.std())

rf_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
rf_pred = rf_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, rf_pred))
print(classification_report(data_class_test, rf_pred))


# 

# 

# # Method 3: Replacing F21 with KN value

# <b> Reading the dataset <b/>

# In[96]:


df = pd.read_csv("CE802_P2_Data.csv")


# <b> Separating the feactures and target <b/>

# In[97]:


df_class = df["Class"]
df = df.drop(columns = ["Class"],axis = 1)


# <b> Replacing F21 missing values with K-Nearest Neighbor imputation method

# In[98]:


df_feat= df.to_numpy()
imputer = KNNImputer(n_neighbors=2, weights="uniform")
df_feat = imputer.fit_transform(df_feat)
df_features = pd.DataFrame(df_feat, index=range(df_feat.shape[0]),
                          columns=range(df_feat.shape[1]))


# <b>Normalization of the features for K-NN and SVM<b/>

# In[99]:


scaler = StandardScaler()
scaler.fit(df_features)
df_features = scaler.transform(df_features)


# <b> Data splitting to verify model accuracy after cross validation and gridsearch <b/>

# In[100]:


data_feat_train, data_feat_test, data_class_train, data_class_test = train_test_split(df_features,df_class,test_size=0.25,stratify=df_class,random_state=1234)


# <b> Pruned Decision Tree <b/>

# In[101]:


clf_tree = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234) #Decision tree with grid search

param_grid = {'max_depth': np.arange(4,21),'min_samples_split': np.arange(4,21),'min_samples_leaf': np.arange(4,21),
              'max_features': ['sqrt','auto','log2']}

tree_gridcv = GridSearchCV(clf_tree,param_grid,cv=10 ,n_jobs=-1)
tree_gridcv.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(tree_gridcv.best_params_))
print("The best score: " + str(tree_gridcv.best_score_))


# In[102]:


clf_tree_prunned = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=1234,
                                               max_depth= tree_gridcv.best_params_['max_depth'],
                                               min_samples_leaf= tree_gridcv.best_params_['min_samples_leaf'],
                                               min_samples_split=tree_gridcv.best_params_['min_samples_split'] )

score_tree = cross_val_score(clf_tree_prunned,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation the whole data set, using the best parameters by gridsearch
print('The average accuracy:', np.mean(score_tree))


clf_tree_prunned.fit(data_feat_train,data_class_train) #compute the confussion matrix by splitting the data into trainning and testing
tree_pred = clf_tree_prunned.predict(data_feat_test)
print(confusion_matrix(data_class_test, tree_pred))
print(classification_report(data_class_test, tree_pred))


# <b> K-Nearest Neighbor (K-NN) <b/>

# In[68]:


knn_gridcv = KNeighborsClassifier()
param_gridsearch = {'n_neighbors': np.arange(1,80),'weights':['uniform','distance']} #dictionary with the number of neighbors to try

knn_gridsearch = GridSearchCV(knn_gridcv,param_gridsearch,cv=10)
knn_gridsearch.fit(data_feat_train,data_class_train)
print("The best parameters: " + str(knn_gridsearch.best_params_))
print("The best score: "+ str(knn_gridsearch.best_score_))


# <B> Cross Validation of K-Nearest Neighbor (K-NN) <b/>

# In[69]:


knn_model = KNeighborsClassifier(n_neighbors = knn_gridsearch.best_params_['n_neighbors'],
                                 weights=knn_gridsearch.best_params_['weights'])

score_knn = cross_val_score(knn_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_knn))

knn_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
knn_pred = knn_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, knn_pred))
print(classification_report(data_class_test, knn_pred))


# <b> Support Vector Machine <b/>

# In[70]:


clf_svm = svm.SVC()
param_grid = {'C': np.logspace(-1, 3, 9),  
              'gamma': np.logspace(-7, -0, 8)}

svm_gridsearch = GridSearchCV(clf_svm,param_grid,n_jobs=-1, cv = 10)
svm_gridsearch.fit(data_feat_train,data_class_train)

print("The best parameters: " + str(svm_gridsearch.best_params_))
print("The best score : " + str(svm_gridsearch.best_score_))


# <b> Cross Validation of Support Vector Machine <b/>

# In[71]:


svm_model = svm.SVC(C = svm_gridsearch.best_params_['C'],gamma=svm_gridsearch.best_params_['gamma'])

score_svm = cross_val_score(svm_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_svm))

svm_model.fit(data_feat_train,data_class_train)#confussion matrix by splitting the data into trainning and testing
svm_pred = svm_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, svm_pred))
print(classification_report(data_class_test, svm_pred))


# <b> Random Forest <b/>

# In[72]:


rf = RandomForestClassifier(criterion='entropy',random_state=1234)
param_grid = {'n_estimators':[400,450,500,550,600],'max_depth': np.arange(4,20)}
#'max_depth': np.arange(4,19),'min_samples_split': np.arange(4,19),'min_samples_leaf': np.arange(4,25)}

rf = GridSearchCV(rf, param_grid,cv=10,n_jobs=-1)
rf.fit(data_feat_train,data_class_train)

print("The best parameters: "+ str(rf.best_params_))
print("The best score: " + str(rf.best_score_))


# <b> Cross Validation of Random Forest <b/>

# In[73]:


rf_model = rf = RandomForestClassifier(criterion='entropy',n_estimators= rf.best_params_['n_estimators'],
                                      max_depth=rf.best_params_['max_depth'],random_state=1234)


score_rf = cross_val_score(rf_model,data_feat_train,data_class_train,cv=10,n_jobs=-1) #cross validation in the whole data set, but with the best parameters by gridsearch
print('The average accuracy:', np.mean(score_rf))
print(score_rf.std())

rf_model.fit(data_feat_train,data_class_train) #confussion matrix by splitting the data into trainning and testing
rf_pred = rf_model.predict(data_feat_test)
print(confusion_matrix(data_class_test, rf_pred))
print(classification_report(data_class_test, rf_pred))


# In[ ]:





# # Prediction on a hold-out test dataset using Forest Tree Classifier

# In[115]:


#Reading the dataset for training the model
data_training = pd.read_csv("CE802_P2_Data.csv")
data_class_train = data_training["Class"]
data_full_train = data_training.drop(columns = ["Class"],axis = 1)

data_full_train.fillna(data_full_train['F21'].mean(),inplace = True)

data_features = imputer.fit_transform(data_full_train)

data_feat = pd.DataFrame(data_features, index=range(data_features.shape[0]),
                                      columns=range(data_features.shape[1])) 
#Normalization
scaler = StandardScaler()
scaler.fit(data_feat)
data_feat = scaler.transform(data_feat)

#Using Forest Tree model to train the dataset
rf_model = rf = RandomForestClassifier(criterion='entropy',n_estimators= 450, max_depth=16, random_state=1234)
rf_model.fit(data_feat,data_class_train)


# In[105]:


#Readng the Test dataset
data_pred= pd.read_csv("CE802_P2_Test.csv")
data_pred = data_pred.drop(columns = ["Class"],axis = 1)

data_full_train.fillna(data_full_train['F21'].mean(),inplace = True)

data_features = imputer.fit_transform(data_pred)

data_pred = pd.DataFrame(data_features, index=range(data_features.shape[0]),
                                      columns=range(data_features.shape[1])) 
#Normalization
scaler = StandardScaler()
scaler.fit(data_pred)
data_pred = scaler.transform(data_pred)


# In[106]:


rf_pred = rf_model.predict(data_pred)

pred =  pd.DataFrame(rf_pred,columns=['Class']) 
final_pred = pd.read_csv('CE802_P2_Test.csv')
final_pred.drop(columns = ['Class'],inplace = True)
submit_csv = pd.concat([final_pred,pred],axis=1)
submit_csv.to_csv("CE802_P2_Test.csv", index = False)


# In[ ]:





# In[ ]:




