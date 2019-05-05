# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:39:30 2019

@author: joshua
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#Load training data
train_load = pd.read_csv('train.csv')

#Load testing data
test_load = pd.read_csv('test.csv')

#Label Data
y_train = train_load['Survived']

#Make the Label Categorical
y_train = y_train.astype('category')
y_train = np.array(y_train)
#Create a variable containing numeric data
train = train_load[['Age',
                    'SibSp',
                    'Parch',
                    'Fare']]
test = test_load[['Age',
                    'SibSp',
                    'Parch',
                    'Fare']]

#Turn the sex category into a binary
sex = train_load['Sex']

sex_dummy = pd.get_dummies(sex)

sex_dummy = sex_dummy.drop('male',
                           axis=1)

sex_dummy = sex_dummy.astype('category')

sextest = test_load['Sex']

sex_dummytest = pd.get_dummies(sextest)

sex_dummytest = sex_dummytest.drop('male',
                           axis=1)

sex_dummytest = sex_dummytest.astype('category')

#Turn passenger class into dummy variables
Pclass = train_load['Pclass']

Pclass_dummy = pd.get_dummies(Pclass)

Pclass_dummy = Pclass_dummy.drop(3,
                                 axis=1)

Pclass_dummy.columns = {'1st Class':1,
                        '2nd Class':2}

Pclass_dummy = Pclass_dummy.astype('category')

Pclasstest = test_load['Pclass']

Pclass_dummytest = pd.get_dummies(Pclasstest)

Pclass_dummytest = Pclass_dummytest.drop(3,
                                 axis=1)

Pclass_dummytest.columns = {'1st Class':1,
                        '2nd Class':2}

Pclass_dummytest = Pclass_dummytest.astype('category')

#Turn embarked into dummy variables
embarked = train_load['Embarked']

embarked_dummy = pd.get_dummies(embarked)

embarked_dummy = embarked_dummy.drop('S',
                                 axis=1)

embarked_dummy = embarked_dummy.astype('category')

embarkedtest = test_load['Embarked']

embarked_dummytest = pd.get_dummies(embarkedtest)

embarked_dummytest = embarked_dummytest.drop('S',
                                 axis=1)

embarked_dummytest = embarked_dummytest.astype('category')

#Merge sex column
train = train.merge(sex_dummy,
                    left_index=True,
                    right_index=True)

test = test.merge(sex_dummytest,
                    left_index=True,
                    right_index=True)

#Merge passenger class columns
train = train.merge(Pclass_dummy,
                    left_index=True,
                    right_index=True)

test = test.merge(Pclass_dummytest,
                    left_index=True,
                    right_index=True)
#Merge embarked columns
train = train.merge(embarked_dummy,
                    left_index=True,
                    right_index=True)

test = test.merge(embarked_dummytest,
                    left_index=True,
                    right_index=True)

train = train.astype('float')
test = test.astype('float')

#Instantiate model
pl = Pipeline([('scaler',RobustScaler(with_centering=True)),
               ('XGBoost',xgb.XGBClassifier())])

param_grid = {"learning_rate": [.001],
              "max_depth": [5],
              "gamma": [1],
              "n_estimators": [100],
              "colsample_bytree": [.7],
              "reg_lambda": [.2],
              "reg_alpha":[.005]}

extreme = XGBClassifier(learning_rate=.001,gamma=0,reg_lambda=.2,reg_alpha=.005)

randomized_mse = GridSearchCV(extreme,param_grid=param_grid,scoring='neg_mean_squared_error',cv=2,verbose=1)
randomized_mse.fit(train,y_train)
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
#print('The root mean square is ' + str(np.mean(np.sqrt(np.abs(cv_results)))))

extreme.fit(train,y_train)
y_preds = extreme.predict(test)

sub = pd.Series(range(0,418))

sub = pd.DataFrame(sub)

sub = sub.merge(pd.DataFrame(y_preds),left_index=True,right_index=True)

sub.columns = {'PassengerId':'O_x','survived':'O_y'}

sub['PassengerId'] = range(892,1310)

submission = sub.to_csv('submission.csv',index=False)

