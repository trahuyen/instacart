# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:03:06 2018

@author: David
"""
#%%
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier, plot_importance

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

target = 'reordered'
IDcol = 'user_id'
#%% Load data and combine
opf = pd.read_csv('order_products__train.csv')
#opp = pd.read_csv('order_products__prior.csv')
#opf = pd.concat([opt, opp])
#del opp
#del opt
orders = pd.read_csv('orders.csv')
df1 = pd.merge(opf, orders, how='left', on='order_id')
del orders
del opf
products = pd.read_csv('products.csv')
df2 = pd.merge(df1, products, how='left', on='product_id')
del products
del df1
department = pd.read_csv('departments.csv')
df3 = pd.merge(df2, department, how='left', on='department_id')
del department
del df2
aisle = pd.read_csv('aisles.csv')
final = pd.merge(df3, aisle, how='left', on='aisle_id')
del aisle
del df3
#%% Move prediction column to last
final=final[['order_id','product_id','add_to_cart_order','user_id','eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order','product_name','aisle_id','department_id','department','aisle','reordered']]
#%% Clean up unnecessary columns
del final['product_name']
del final['aisle']
del final['department']
del final['eval_set']
#%%
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['reordered'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['reordered'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['reordered'], dtrain_predprob))
                    
    plot_importance(alg)
    plt.show()
#%% Choose all predictors except target & IDcols
%%time
predictors = [x for x in final.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, final, predictors)
#%% First Parameter test
%%time
param_test1 = {
    'max_depth':list(range(3,10,2)),
    'min_child_weight':list(range(1,6,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch1.fit(final[predictors],final[target])
#%%
%%time
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#%% Second parameter test
%%time
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[2,3,4]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=9, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch2.fit(final[predictors],final[target])
#%%
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#%%
%%time
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=4,
                                        min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch3.fit(final[predictors],final[target])
#%%
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#%%
%%time
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=4,
 min_child_weight=4, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch6.fit(final[predictors],final[target])
#%%
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
#%%
%%time
param_test7 = {
 'reg_alpha':[100,200]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=4, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch7.fit(final[predictors],final[target])
#%%
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
#%%
xgb4 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=177,
 max_depth=4,
 min_child_weight=4,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=100,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, final, predictors)