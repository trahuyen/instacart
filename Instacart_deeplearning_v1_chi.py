# PROJECT DESCRIPTION
#
# In this competition, Instacart is challenging the Kaggle community to use this anonymized data on customer orders over time to predict which previously purchased products will be in a user’s next order. They’re not only looking for the best model, Instacart’s also looking for machine learning engineers to grow their team.

#========================================================================================
from scipy import stats
import numpy as np
import pandas as pd
# %matplotlib inline

input_link='C:/Users/sentifi/kaggle/competitions/instacart-market-basket-analysis/'
orders = pd.read_csv(input_link + 'orders_csv/orders.csv', encoding='utf-8')
prior = pd.read_csv(input_link +'order_products__prior_csv/order_products__prior.csv')
# department = pd.read_csv(input_link +'departments_csv/departments.csv')
# training = pd.read_csv(input_link +'order_products__train_csv/order_products__train.csv')
# aisles = pd.read_csv(input_link +'aisles_csv/aisles.csv')
product = pd.read_csv(input_link +'products_csv/products.csv')

# Clustering
# what we want to find out: which previously purchased products will be in a user’s next order
#     1. What factors influence user's next orders?
#         1.1. Old demand group
#             1.1.1. Demand in sequence timing
#             1.1.2. Income (how often they order)
#             1.2.3. Product deadline
#             1.2.4. Promotion
#             1.2.5. Job busy or not ( time order, how often order, which day order)
#         1.2. New demand group
#             1.2.1. Curiosity - 'Attempted to new'
#             1.2.2. Follow the trend - 'buy the new and popular product at the moment'
#             1.2.3. Promotion

# ->> we want to find out similarity among users: who's belong to old demand group/ new demand group  , group base on income/ job busy or time order...
# ->> we want to find out similarity of products/department/hours that have high reorder ratio

# Prepare data
df1 = pd.merge(prior, orders, how='left', on='order_id')
training_set = pd.merge(df1,product, how='left', on='product_id')

del orders
del product
del df1
del prior

# Building features
# Build features and # one_hot encoding: convert categories to numberic if necessary

print('product related features')
pro=pd.DataFrame()
pro['orders'] = training_set.groupby('product_id').size().astype(np.int32)
pro['reorders']= training_set.groupby(['product_id'])['reordered'].sum()
pro['reorder_rate'] = (pro.reorders / pro.orders).astype(np.float32)
pro=pro.reset_index()


print('computing user features')
usr=pd.DataFrame()
usr['total_items'] = training_set.groupby('user_id')['product_id'].size().astype(np.int16)
usr['total_orders'] = training_set.groupby('user_id')['order_id'].nunique() # Count unique
usr['days_since_prior_order_average'] = training_set.groupby('user_id')['days_since_prior_order'].mean().astype(np.int16)
usr['order_hour_of_day_median'] = training_set.groupby('user_id')['order_hour_of_day'].median().astype(np.int16)
usr['order_dow_mode'] = training_set.groupby('user_id')['order_dow'].apply(lambda x: tuple(stats.mode(x)[0])[0])
usr=usr.reset_index()

# training_set['all_products'] = training_set.groupby('user_id')['product_id'].apply(set)

# Combine all features
final=pd.merge(pro,training_set,on='product_id',how='right')
final1=pd.merge(usr,training_set,on='user_id',how='right')


# Other features

final1['average_basket'] = (final1.total_items/final1.total_orders).astype(np.float32)
final1['days_since_ratio'] = final1.days_since_prior_order / final1.days_since_prior_order_average

del pro
del usr
del final

# Detect null value
# nul_col=final1.columns[final1.isnull().values.any()]
# for i in nul_col:
#     print(final1[i].isnull().sum())

final1=final1.fillna(0,inplace=False)
final1=final1.replace([np.inf,-np.inf],0)
final1=final1.drop(['product_name','eval_set','add_to_cart_order'],axis=1)

# Change column position.
cols = list(final1.columns.values) # Make a list of all of the columns in the training_set
cols.pop(cols.index('product_id'))# Remove product_id from list
cols.pop(cols.index('order_id'))
cols.pop(cols.index('reordered'))
cols.pop(cols.index('user_id'))
final1 = final1[['product_id']+['order_id']+['user_id']+['reordered']+cols] # Add product_id to list

X = final1.values[:,4:].astype(np.int32)
Y = final1.values[:,3].astype(np.int32)

# Correlation Matrix
import seaborn as sns

corr = final1.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)


from sklearn.preprocessing import StandardScaler
# Way 1: Variance threshold: removing features with low variance . Select features before training
from sklearn.feature_selection import VarianceThreshold
selection_thres= VarianceThreshold(threshold=(.3 * (1 - .3)))  # 30%
X2=selection_thres.fit_transform(X)

# Way 2: Use Singular value decomposition to reduce dimension before input in the training model
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=8, n_iter=5, random_state=0)
sc = StandardScaler()
X_scaler=sc.fit_transform(X)
X=svd.fit_transform(X_scaler)

# limitation of using this method is we don't know which features they keep to label features to interpret which features are important


# Way 3: L1-based feature selection to select the non-zero coefficients.
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc=LinearSVC(C=0.01, penalty='l1',dual=False).fit(X,Y)
model =SelectFromModel(lsvc,prefit=True)
X=model.transform(X)
print(svc.coef_) - #check which column = 0 -> mean ignore
#or print(model.get_support())


# Way 4: Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
sc = StandardScaler()
X=sc.fit_transform(X)
clf1=ExtraTreesClassifier()
clf1=clf1.fit(X,Y)
model = SelectFromModel(clf1, prefit=True)
X = model.transform(X)

#==================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB as gnb # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier as knn # K-NN
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3,random_state=10)

# %%time
# Spot Check Algorithms
models = []
# models.append(('LDA', lda()))
# models.append(('KNN', knn()))
# models.append(('NB',gnb()))
# # models.append(('HMM', ghmm()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []

# %%time
for name, model in models:
    kfold= KFold(n_splits=10, random_state=42)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='f1') # different scoring 'precision','recall','f1','roc_auc' for binary targets only
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Experiment with Leave One OUt Cross Validation

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#==================================================
from sklearn.ensemble import RandomForestClassifier as rf
clf = rf(n_jobs=1, random_state=10, n_estimators= 100,bootstrap=True, class_weight=None).fit(x_train,y_train)
y_pred=clf.predict(x_test)


#==================================================
# Gradient Boosted Tree
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

param_grid = {'max_depth': [4,6,8],
          'learning_rate': [0.1,0.05,0.02,0.01],
          'loss': 'huber','alpha':0.95,
          'min_samples_leaf':[3,5,7]}
params = {'max_depth': 6,
          'learning_rate': 0.05,
          'loss': 'huber','alpha':0.95,
          'min_samples_leaf':5,'random_state':42}
clf = GradientBoostingRegressor(**params).fit(x_train, y_train)

# Way 5: feature selection is used as pre-processing step before doing the actual learning
# from sklearn.pipeline import Pipeline
# clf = Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#   ('classification', GradientBoostingRegressor())
# ])
# clf.fit(X, y)

# choose best param with grid_search
est=GradientBoostingRegressor(n_estimators=500)
gs=GridSearchCV(est,param_grid,n_job=1).fit(x_train, y_train)
y_pred2=gs.predict(x_test)
y_pred_final=y_pred2.round().astype(int)
mse = mean_squared_error(y_test, y_pred_final)
r2 = r2_score(y_test, y_pred_final)
f1_score(y_test,y_pred_final)

y_pred2=clf.predict(x_test)

print( "MSE: %.4f" %mse)
print("r2: %.4f" %r2 )

features = final1.columns[4:]
weight = list(clf.feature_importances_)
importance = zip(features, weight)
importance = sorted(importance, key=lambda x: x[1])
total = sum(j for i, j in importance)
importance2 = [(i, float(j)/total) for i, j in importance]
print(importance2)

import matplotlib.pyplot as plt
# unzip importance features to draw graph
feat_name, wei=zip(*importance)
plt.barh(np.arange(len(features))+ 0.25, wei)

plt.yticks(np.arange(len(features)) + 0.25, feat_name)

_ = plt.xlabel('Relative importance')

from sklearn.ensemble.partial_dependence import plot_partial_dependence

fig, axs = plot_partial_dependence(clf, x_train,
                                   features=[5,10,9,11],
                                   feature_names=features,
                                   n_cols=2)

fig.show()

# use feature importance for feature selection
from sklearn.feature_selection import SelectFromModel

# Way 2:
from sklearn.svm import LinearSVC

# Fit model using each importance as a threshold
thresholds = list(clf.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(clf, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train)
	# train model
	selection_model = GradientBoostingRegressor()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(x_test)
	y_pred4 = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred4]
	accuracy = f1_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

#######################################
# Lightgbm

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

train_data = lgb.Dataset (x_train, label=y_train)
param = {'num_leaves': 3, 'max_depth': 7,'learning_rate': .05, 'max_bin': 200}
param['metric'] = ['auc', 'multi_logloss']
# training our model using light gbm
num_round = 30
lgbm = lgb.train (param,train_data, num_round)
ypred = lgbm.predict(x_test)

mse = mean_squared_error(y_test, ypred)
r2 = r2_score(y_test, ypred)

print( "MSE: %.4f" %mse)
print("r2: %.4f" %r2 )


#######################################
# Save model
import pickle
filename = 'finalized_model_v1_chi.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(x_test)
print(result)

result2=result.round().astype(int)
mse = mean_squared_error(y_test, result2)
r2 = r2_score(y_test, result2)
f1_score(y_test,result2)


################################
# APPLY DEEP LEARNING
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# Keras experiments
from keras.models import Sequential
from keras.layers import Dense
def model():
    model=Sequential()
    model.add(Dense(unit=32, activation='relu',input_dim=8))
    model.add(Dense(unit=16, activation='relu'))
    model.add(Dense(unit=8, activation='relu'))
    model.add(Dense(unit=1, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=32, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


###########################################
# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

model.fit(x_train, y_train, epochs=50, batch_size=32)
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)

classes = model.predict(x_test, batch_size=128)

model.summary()


#================================================
# to prevent overfitting, we need to create data. While collecting data mining might be expensive
# we can try to do data augmentation


