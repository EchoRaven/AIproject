# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:16:45 2022

@author: thb
"""
import pandas as pd
from sklearn import datasets
wine=datasets.load_wine()
wine_data=wine.data
wine_target=wine.target
wine_data=pd.DataFrame(data=wine_data)
wine_target=pd.DataFrame(data=wine_target)

wine_data.insert(0,'class',wine_target)
wine=wine_data
features=wine.drop(columns=['class'],axis=1)
targets=wine['class']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,targets,test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
x_train_scaled=MM.fit_transform(x_train)
x_test_scaled=MM.transform(x_test)

#使用多种回归方式
#SVC回归
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

tuned_parameters=[
    {"kernel":["rbf"],"gamma":[1,0.1,1e-2,1e-3,1e-4],"C":[1,10,100,1000]},
    {"kernel":["linear"],"C":[1,10,100,1000]},
    {"kernel":["poly"],"gamma":[10,1],"C":[1,10],"degree":[2,3,4,5,6]}]

clf1=GridSearchCV(SVC(),tuned_parameters)
clf1.fit(x_train_scaled,y_train)

print(f"SVC最优超参数组合:{clf1.best_params_}")
print(f"SVC最优模型交叉验证得分:{clf1.best_score_:.4f}")

#KNeighbor回归
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(algorithm="ball_tree")
param_grid = [
    {
        "weights":["uniform"],
        "n_neighbors":[i for i in range(1,11)]
    },
    {
        "weights":["distance"],
        "n_neighbors":[i for i in range(1,11)],
        "p":[i for i in range(1,10)]
    }]

clf2=GridSearchCV(KNeighborsClassifier(),param_grid)
clf2.fit(x_train_scaled,y_train)

print(f"KN最优超参数组合:{clf2.best_params_}")
print(f"KN最优模型交叉验证得分:{clf2.best_score_:.4f}")

#LogisticRegression
from sklearn.linear_model import LogisticRegression
Log_param=[{"max_iter":[500,1000,10000,100000],"penalty":["none"],"tol":[1e-4,1e-3,1e-2,0,1,1,10,100]},
           {"max_iter":[500,1000,10000,100000],"penalty":["l2"],"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],"tol":[1e-4,1e-3,1e-2,0,1,1,10,100]}]
clf3=GridSearchCV(LogisticRegression(),Log_param)
clf3.fit(x_train_scaled,y_train)

print(f"Log最优超参数组合:{clf3.best_params_}")
print(f"Log最优模型交叉验证得分:{clf3.best_score_:.4f}")

#Ridge
from sklearn import linear_model
Ridge_param=[
    {"solver":['auto','svd','cholesky','sparse_cg','lsqr','sag'],
     "alpha":[0,1e-1,1e-2,1e-3,1e-4],"max_iter":[500,1000,10000],
     "tol":[1e-4,1e-3,1e-2,0.1]}]
clf4=GridSearchCV(linear_model.Ridge(),Ridge_param)
clf4.fit(x_train_scaled,y_train)

print(f"Log最优超参数组合:{clf4.best_params_}")
print(f"Log最优模型交叉验证得分:{clf4.best_score_:.4f}")