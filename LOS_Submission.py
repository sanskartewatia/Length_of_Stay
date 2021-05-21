#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 
import random
from sklearn import preprocessing
from math import sqrt
from numpy import mean
from numpy import std
from sklearn.metrics import make_scorer, accuracy_score, r2_score
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

train = pd.read_csv("dataset_1.csv")
train.head(10)


# In[2]:


sns.boxplot(x=train['LOS'])


# In[3]:


train.nunique()


# In[5]:


from sklearn.cluster import KMeans
import plotly.graph_objects as go

sse = []
target = train['LOS'].to_numpy().reshape(-1,1)
num_clusters = list(range(1, 10))

for k in num_clusters:
    km = KMeans(n_clusters=k)
    km.fit(target)
    sse.append(km.inertia_)

fig = go.Figure(data=[
    go.Scatter(x = num_clusters, y=sse, mode='lines'),
    go.Scatter(x = num_clusters, y=sse, mode='markers')
])

fig.update_layout(title="Evaluation on number of clusters:",
                 xaxis_title = "Number of Clusters:",
                 yaxis_title = "Sum of Squared Distance",
                 showlegend=False)
fig.show()


# In[8]:


import plotly.express as px 
km = KMeans(3)
km.fit(train['LOS'].to_numpy().reshape(-1,1))
train.loc[:,'Day labels'] = km.labels_
fig = px.scatter(train,'LOS', 'injury', color='Day labels')
fig.update_layout(title = "clusters.",
                 xaxis_title="injury", yaxis_title="LOS")
fig.show()


# In[22]:


# Creating a barplot of 'Type of Respiratory Issue vs LOS'
plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['respiratory'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of Respiratory issue', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[23]:


plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['skin'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of Skin Disease', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[7]:


plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['nervous'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of nervous issue', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[16]:


plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['injury'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of  injury', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[18]:


plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['mental'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of mental state', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[19]:


plt.figure(figsize = (25, 15))
plt.style.use('seaborn')
sns.barplot(train['congenital'], train['LOS'], palette = 'Paired')
plt.xlabel('Type of congenital disease', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[24]:


plt.figure(figsize = (10, 8))
plt.style.use('seaborn')
sns.barplot(train['ADM_URGENT'], train['LOS'], palette = 'Paired')
plt.xlabel('Urgent Admission', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[30]:


plt.figure(figsize = (10,8))
plt.style.use('seaborn')
sns.barplot(train['INS_Self Pay'], train['LOS'], palette = 'Paired')
plt.xlabel('Bills Paid by Self', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[34]:


plt.figure(figsize = (10,8))
plt.style.use('seaborn')
sns.barplot(train['INS_Medicaid'], train['LOS'], palette = 'Paired')
plt.xlabel('Bills Paid by Medical Aid Insurance', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.ylabel('LOS', fontdict = { 'fontsize' : 25, 'fontweight' : 'bold'})
plt.tick_params(labelsize = 25)
plt.show()


# In[4]:


top = np.percentile(train.LOS,99.8)
train = train[ (train['LOS']<top) ]
sns.boxplot(x=train['LOS'])


# In[5]:


corr=train.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[6]:


train2=train.drop(columns=['ADM_EMERGENCY'])
train2=train2.drop(columns=['MAR_UNKNOWN (DEFAULT)'])
train2=train2.drop(columns=['NICU'])
train2=train2.drop(columns=['AGE_newborn'])
train2=train2.drop(columns=['ETH_WHITE'])
train2=train2.drop(columns=['REL_UNOBTAINABLE'])
train2=train2.drop(columns=['AGE_senior'])

corr2=train2.corr()
plt.figure(figsize=(18,18))
sns.heatmap(corr2, 
        xticklabels=corr2.columns,
        yticklabels=corr2.columns)


# In[7]:


train2.LOS.describe()


# In[8]:


train2.nunique()


# In[9]:


train2.info()


# In[61]:


insurance = train2[['INS_Government', 'INS_Medicaid','INS_Medicare', 'INS_Private','INS_Self Pay']].copy()
boxplot = insurance.boxplot()


# In[10]:


Y = train2['LOS'].values
X = train2.drop(columns=['LOS'])
print("Total Dataset Shape - ", X.shape)
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Train Set shape - ", X_train.shape)
print("Test Set shape - ", X_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LinearRegression()
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores), std(scores)))
print(-1*scores)
model.fit(X_train, y_train)
pred=model.predict(X_test);
mse = mean_squared_error(y_test, pred)
r2s = r2_score(y_test, pred)
print("Performance on test set - ")
print("MSE: %f" % (mse))
print("R2: %f" % (r2s))


# In[ ]:


from sklearn.linear_model import Lasso

m3 = Lasso()
params = {'alpha': [0.4, 0.6, 0.8, 1] }
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m3, param_grid=params, scoring=scorer, cv=5)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[ ]:


from sklearn.linear_model import Lasso
cv3 = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model3 = Lasso(alpha=1)
# evaluate model
scores3 = cross_val_score(model3, X_train, y_train, scoring='neg_mean_squared_error', cv=cv3, n_jobs=-1)

# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores3), std(scores3)))
print(-1*scores3)
model3.fit(X_train, y_train)
pred3=model3.predict(X_test);
mse3 = mean_squared_error(y_test, pred3)
r2s3 = r2_score(y_test, pred3)
print("Performance on test set - ")
print("MSE: %f" % (mse3))
print("R2: %f" % (r2s3))


# In[ ]:


from sklearn.linear_model import Ridge
m4 = Ridge()
params = {'alpha': [0.2, 0.4, 0.6, 0.8, 1] }
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m4, param_grid=params, scoring=scorer, cv=10)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[ ]:


from sklearn.linear_model import Ridge
cv4 = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model4 = Ridge(alpha=0.2)
# evaluate model
scores4 = cross_val_score(model4, X_train, y_train, scoring='neg_mean_squared_error', cv=cv4, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores4), std(scores4)))
print(-1*scores4)
model4.fit(X_train, y_train)
pred4=model4.predict(X_test);
mse4 = mean_squared_error(y_test, pred4)
r2s4 = r2_score(y_test, pred4)
print("Performance on test set - ")
print("MSE: %f" % (mse4))
print("R2: %f" % (r2s4))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
m5 = KNeighborsRegressor(n_jobs=-1)
params = {'n_neighbors': [5,7,11,15], 'p':[1,2], 'leaf_size':[5,15,25,30,45]}
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m5, param_grid=params, scoring=scorer, cv=10)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
cv5 = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model5 = KNeighborsRegressor(leaf_size= 5, n_neighbors= 5, p= 1,n_jobs=-1)
# evaluate model
scores5 = cross_val_score(model5, X_train, y_train, scoring='neg_mean_squared_error', cv=cv5, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores5), std(scores5)))
print(-1*scores5)
model5.fit(X_train, y_train)
pred5=model5.predict(X_test);
mse5 = mean_squared_error(y_test, pred5)
r2s5 = r2_score(y_test, pred5)
print("Performance on test set - ")
print("MSE: %f" % (mse5))
print("R2: %f" % (r2s5))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
m6 = DecisionTreeRegressor()
params = {'max_depth': [6,8,10], 'min_samples_leaf':[1,5,10,20,50]}
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m6, param_grid=params, scoring=scorer, cv=10)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model6 = DecisionTreeRegressor(random_state=0,max_depth = 10, min_samples_leaf = 1)
cv6 = KFold(n_splits=10, random_state=1, shuffle=True)
# evaluate model
scores6 = cross_val_score(model6, X_train, y_train, scoring='neg_mean_squared_error', cv=cv6, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores6), std(scores6)))
print(-1*scores6)
model6.fit(X_train, y_train)
pred6=model6.predict(X_test);
mse6 = mean_squared_error(y_test, pred6)
r2s6 = r2_score(y_test, pred6)
print("Performance on test set - ")
print("MSE: %f" % (mse6))
print("R2: %f" % (r2s6))

importance = model6.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,100*v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], 100*importance)
pyplot.show()


# In[ ]:


import xgboost as xgb
m7 = xgb.XGBRegressor(random_state=0,n_jobs=-1,tree_method='gpu_hist', gpu_id=0)
params = {'max_depth': [10],'eta': [0.05, 0.1], 'lambda': [0.9, 1], 'alpha': [0.1,0.02] }
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m7, param_grid=params, scoring=scorer, cv=3)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[ ]:


model7 = xgb.XGBRegressor(random_state=0,n_jobs=-1,tree_method='gpu_hist', gpu_id=0, alpha= 0.1, eta= 0.1, max_depth= 10)
cv7 = KFold(n_splits=10, random_state=1, shuffle=True)
# evaluate model
scores7 = cross_val_score(model7, X_train, y_train, scoring='neg_mean_squared_error', cv=cv7, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores7), std(scores7)))
print(-1*scores7)
model7.fit(X_train, y_train)
pred7=model7.predict(X_test);
mse7 = mean_squared_error(y_test, pred7)
r2s7 = r2_score(y_test, pred7)
print("Performance on test set - ")
print("MSE: %f" % (mse7))
print("R2: %f" % (r2s7))

importance = model7.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,100*v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], 100*importance)
pyplot.show()


# In[ ]:


import h2o
from h2o.automl import H2OAutoML
h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=6  # in gigabytes
)


# In[ ]:


train_h2o=pd.DataFrame(data=X_train)
train_h2o['LOS']=y_train
#X_train, X_test, y_train, y_test
h20_train=h2o.H2OFrame(train_h2o)
x1 = h20_train.columns
y1 = "LOS"
x1.remove(y1)
X_test_test= h2o.H2OFrame(X_test)
print('done')


# In[ ]:





# In[ ]:


aml = H2OAutoML(max_models=12, seed=1)
aml.train(x=x1, y=y1, training_frame=h20_train)


# In[ ]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[ ]:


predleader = aml.leader.predict(X_test_test)
pred = aml.predict(X_test_test)


# In[ ]:


dfpred = h2o.as_list(predleader)
mse8 = mean_squared_error(y_test, dfpred)
r2s8 = r2_score(y_test, dfpred)


# In[ ]:


print("Performance on test set - ")
print("MSE: %f" % (mse8))
print("R2: %f" % (r2s8))


# In[ ]:


h2o.shutdown()


# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle


# In[ ]:


Y = train2['LOS'].values
X = train2.drop(columns=['LOS'])
print("Total Dataset Shape - ", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Train Set shape - ", X_train.shape)
print("Test Set shape - ", X_test.shape)


# In[ ]:


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) # Output
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='mean_squared_error')
estp = EarlyStopping(monitor='val_loss', min_delta=0,patience=5, verbose=1, mode='auto',restore_best_weights=True)
model.fit(X_train,y_train,validation_split=0.15,shuffle='True',verbose=2,epochs=200, callbacks=[estp])


# In[ ]:


model.summary()


# In[ ]:


pred9 = model.predict(X_test)
mse9 = mean_squared_error(y_test, pred9)
r2s9 = r2_score(y_test, pred9)


# In[ ]:


print("Performance on test set - ")
print("MSE: %f" % (mse9))
print("R2: %f" % (r2s9))


# In[12]:


from sklearn.ensemble import RandomForestRegressor
m6 = RandomForestRegressor()
params = {'max_depth': [8,10], 'min_samples_leaf':[5,10,20], 'n_jobs':[-1] }
scorer = make_scorer(mean_squared_error)
clf_grid = GridSearchCV(estimator=m6, param_grid=params, scoring=scorer, cv=10)

clf_grid.fit(X_train, y_train)
best_param = clf_grid.best_params_
best_param


# In[17]:


model9 = RandomForestRegressor(random_state=0,n_jobs=-1, max_depth= 8, min_samples_leaf= 20)
cv9 = KFold(n_splits=10, random_state=1, shuffle=True)
# evaluate model
scores9 = cross_val_score(model9, X_train, y_train, scoring='neg_mean_squared_error', cv=cv9, n_jobs=-1)
# report performance
print("Cross-Validatoin Performance - ")
print('Avergae MSE: (%.3f), Std: (%.3f)' % (-1*mean(scores9), std(scores9)))
print(-1*scores9)
model9.fit(X_train, y_train)
pred9=model9.predict(X_test);
mse9 = mean_squared_error(y_test, pred9)
r2s9 = r2_score(y_test, pred9)
print("Performance on test set - ")
print("MSE: %f" % (mse9))
print("R2: %f" % (r2s9))


# In[ ]:




