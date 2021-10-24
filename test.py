#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
data_train = pd.read_excel('./data.xlsx')
# %%
# 获取数据
cols = ['usage','source', 'decoration','distance', 'area', 'days']
x = data_train[cols].values
y = data_train['single_price'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
# %%
clfs = {
        'svm':svm.SVR(), 
        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
        'BayesianRidge':linear_model.BayesianRidge()
       }
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        y_pred = clfs[clf].predict(X_test)
        print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )
    except Exception as e:
        print(clf + " Error:")
        print(str(e))
# %%
cols = ['usage','source', 'decoration','distance', 'area', 'days']
x = data_train[cols].values
y = data_train['single_price'].values
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
# %%
sum(abs(y_pred - y_test))/len(y_pred)
# %%
rfr = clf
data_test = pd.read_excel('./predict.xlsx')
data_test[cols].isnull().sum()
# %%
data_test['area'].describe()
# %%
cols2 = ['usage','source', 'decoration','distance', 'area', 'days']
data_test_x = pd.concat( [data_test[cols2]] ,axis=1)
data_test_x.isnull().sum()
# %%
x = data_test_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)

print(y_te_pred.shape)
print(x.shape)
# %%
prediction = pd.DataFrame(y_te_pred, columns=['single_price'])
result = pd.concat([ data_test['title'], prediction], axis=1)
# result = result.drop(resultlt.columns[0], 1)
result.columns
# %%
result.to_excel('./Predictions.xlsx', index=False)
# %%
