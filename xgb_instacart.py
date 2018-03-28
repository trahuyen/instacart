# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:15:01 2018

@author: David
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_selection import f_regression, mutual_info_regression
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import itertools
import pickle
from scipy import integrate
#%%
def capcurve(y_test, y_pred_proba):
    num_pos_obs = np.sum(y_test)
    num_count = len(y_test)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)
    y_cap = np.c_[y_test,y_pred_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(drop=True)
    
    print(y_cap_df_s.head(20))
    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0
    
    percent = 0.5
    row_index = int(np.trunc(num_count * percent))
    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)
    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy,xx)
    sigma_random = integrate.simps(xx,xx)
    
    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    #ar_label = 'ar value = %s' % ar_value
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')
    #ax.scatter(xx,yy, color='red')
    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')
    
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("CAP Curve - a_r value ="+str(ar_value))
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    plt.show()
#%% Load data and combine
opt = pd.read_csv('order_products__train.csv')
opp = pd.read_csv('order_products__prior.csv')
opf = pd.concat([opt, opp])
del opp
del opt
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
#%% Split into regressors and predicted
X = final.iloc[:,:10]
Y = final.iloc[:,10]
del final
#%% Split data into train and test sets
seed = 42
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#%% Training the model
%%time
model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=4,
 min_child_weight=4,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=100,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)
#%% Fitting the model
%%time
model.fit(X_train, y_train)
#%% Evaluate model with KFold cross validation
%%time
kfold = KFold(n_splits=10, random_state=42)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#%% Show feature importance of regressors
plot_importance(model)
plt.show()
#%% Make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#%%
y_pred_proba = model.predict_proba(X_test)
#%%
capcurve(y_test, y_pred_proba[:,1])
#%% Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#%% Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
thresh = cm.max() / 2.
fmt = 'd'
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
#%% ROC
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#%% MSE
mse = mean_squared_error(y_test, predictions)
print(mse)
#%%
r2 = r2_score(y_test, predictions)
print(r2)
#%% Save model
pickle.dump(xgb4, open('test.pickle.dat', 'wb'))
#%% Load model
xgb4 = pickle.load(open("test.pickle.dat", "rb"))
#%%
f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
