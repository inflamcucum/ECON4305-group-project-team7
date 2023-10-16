#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Standard libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

#Sklearn
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


# In[2]:


data = pd.read_excel('Downloads/US FRED-MD Macro Dataset.xlsx', 'US FRED-MD Macro Data',header=0, index_col=0, parse_dates=True)
tran_code = pd.read_excel('Downloads/US FRED-MD Macro Dataset.xlsx', 'Transformation Code',header=0, index_col=0, parse_dates=True)


# In[3]:


data


# In[4]:


tran_code


# In[5]:


def data_transform(x,tran_code):
    if tran_code == 1:
        x_tr = x        
        
    elif tran_code == 2:
        x_tr = x.diff()
            
    elif tran_code == 3:
        x_tr = x.diff().diff()
        
    elif tran_code == 4:
        x_tr = np.log(x)
        
    elif tran_code == 5:
        x_tr = np.log(x).diff()*100
    
    elif tran_code == 6:
        x_tr = np.log(x).diff().diff()*100
    
    elif tr
        x_tr = (x.pct_change()-1)*100
    
    return x_tr 


# In[88]:


macro_tr=[]

for col in data.columns:
    tr_code = tran_code[col].values
    data_tr = data_transform(data[col], tr_code)
    macro_tr.append(data_tr)

macro_tr = pd.DataFrame(macro_tr).T

#We use the data starting from 1992-03-01 since older data are imcomplete
macro_final = macro_tr.iloc[396:,:]
macro_final = macro_final.dropna()
macro_final


# In[7]:


corr = macro_final.iloc[:,1:].corr()
top_corr_features = corr.index[abs(corr["CPIAUCSL"])>=0.1]
bottom_corr_features = corr.index[abs(corr["CPIAUCSL"])<0.1]

corr


# In[8]:


top_corr_features


# In[9]:


bottom_corr_features


# In[10]:


macro_final = macro_final.drop(columns=bottom_corr_features)
macro_final


# In[11]:


y = macro_final['CPIAUCSL']


# In[12]:


X = macro_final.drop(columns=['CPIAUCSL'])
X


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[14]:


def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r_squared = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R-squared', r_squared)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r_squared = metrics.r2_score(true, predicted)
    return mae, rmse, r_squared


# In[15]:


lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
pred = lin_reg.predict(X_test)

train_pred = lin_reg.predict(X_train)
test_pred = lin_reg.predict(X_test)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
print('====================================')
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)


# In[16]:


coeff_lin = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Linear: Coef'])
coeff_lin


# In[17]:


plt.scatter(y_test, pred)
plt.plot(y_test,y_test,'k-') # identity line


# In[18]:


results_df_lin = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred)]], 
                          columns=['Model', 'MAE', 'RMSE', 'R-squared'])
results_df_lin


# In[19]:


#Ridge regression
ridge = Ridge(alpha=1000)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)

train_pred = ridge.predict(X_train)
test_pred = ridge.predict(X_test)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
print('====================================')
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)


# In[20]:


coeff_ridge = pd.DataFrame(ridge.coef_, X.columns, columns=['Ridge: Coef'])
pd.concat([coeff_lin, coeff_ridge],axis=1)


# In[21]:


plt.scatter(y_test, pred)
plt.plot(y_test,y_test,'k-') # identity line


# In[22]:


results_df_ridge = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'RMSE', 'R-squared'])
results_df_ridge

pd.concat([results_df_lin, results_df_ridge], axis=0, ignore_index=True)


# In[23]:


#LASSO regression
lasso = Lasso(alpha=1000)
lasso.fit(X_train, y_train)

train_pred = lasso.predict(X_train)
test_pred = lasso.predict(X_test)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
print('====================================')
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)


# In[24]:


coeff_lasso = pd.DataFrame(lasso.coef_, X.columns, columns=['Lasso: Coef'])
pd.concat([coeff_lin, coeff_ridge, coeff_lasso],axis=1)


# In[25]:


plt.scatter(y_test, pred)
plt.plot(y_test,y_test,'k-') # identity line


# In[26]:


results_df_lasso = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'RMSE', 'R-squared'])

pd.concat([results_df_lin, results_df_ridge, results_df_lasso], axis=0, ignore_index=True)


# In[27]:


#Elastic net 
elastic = ElasticNet(alpha=100, l1_ratio=0.5, max_iter=30000)
elastic.fit(X_train, y_train)

test_pred = elastic.predict(X_test)
train_pred = elastic.predict(X_train)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
print('====================================')
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)


# In[28]:


coeff_elastic = pd.DataFrame(elastic.coef_, X.columns, columns=['Elastic Net: Coef'])
pd.concat([coeff_lin, coeff_ridge, coeff_lasso, coeff_elastic],axis=1)


# In[29]:


plt.scatter(y_test, pred)
plt.plot(y_test,y_test,'k-') # identity line


# In[30]:


results_df_elastic = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'RMSE', 'R-squared'])

pd.concat([results_df_lin, results_df_ridge, results_df_lasso, results_df_elastic], axis=0, ignore_index=True)


# In[31]:


X_start = 1
X_lag = 1


X = X.iloc[X_start:,:].shift(X_lag).dropna()
X


# In[85]:


y = y.drop(y.index[:2])
y


# In[33]:


from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2


# In[34]:


#PCA
steps_X = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
           ('pca', PCA(n_components = 1, random_state=1))]

pl_pca_X = Pipeline(steps_X)

X_pca_model = pl_pca_X.fit(X)

print('n_components:', 1, '  ', 'variance explained:', '%.3f' %X_pca_model.steps[1][1].explained_variance_ratio_.sum())

X_pca = X_pca_model.transform(X)


# In[35]:


for n_comp in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30]:

    steps_X = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
               ('pca', PCA(n_components = n_comp, random_state=1))]
    
    pl_pca_X = Pipeline(steps_X)
    X_pca_model = pl_pca_X.fit(X)
    
    print('n_components:', n_comp, '  ', 'variance explained:', '%.3f' %X_pca_model.steps[1][1].explained_variance_ratio_.sum())


# In[36]:


n_train = 25
n_records = len(X)
n_forecast = n_records-n_train

j = 0
for i in range(n_train, n_records):    
    #train, test = X[0:i], X[i:i+1]   #expanding window
    train, test = X[j:i], X[i:i+1]   #rolling window
    print('train=%d, test=%d' % (len(train), len(test)))
    j += 1


# In[37]:


def pipeline(config):
    n_comps = config
    steps = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
             ('pca', PCA(n_components = n_comps, random_state=1)),
             ('ols', LinearRegression())   
            ]
    pipeline = Pipeline(steps)
    return pipeline


# In[38]:


def walk_forward_validation(cfg):
    
    n_train = 25

    n_records = len(X)

    y_test_list = []
    
    y_pred_list = []

    j = 0

    for i in range(n_train, n_records):
    
        X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
       
        model = pipeline(cfg).fit(X_train, y_train)        
    
        y_pred = model.predict(X_test)   #Predicted GDP Growth
    
        y_pred_list.extend(y_pred)
    
        y_test_list.extend(y_test)

        j += 1
    
    score_rmse = metrics.mean_squared_error(y_test_list, y_pred_list, squared=False)
    
    print(' > %.3f' % score_rmse)
    
    return score_rmse 


# In[39]:


# score a model, return None on failure
def repeat_evaluate(config, n_repeats=1):
    # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(config) for _ in range(n_repeats)]
    # summarize score
    result = np.mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# In[40]:


# grid search configs
def grid_search(cfg_list):
    # evaluate configs
    scores = [repeat_evaluate(cfg) for cfg in cfg_list]
    # sort configs by score_rmse, asc
    #scores.sort(key=lambda tup: tup[1])
    scores.sort(key=lambda tup: tup[1], reverse=True)
    return scores


# In[41]:


# create a list of configs to try
def model_configs():
    # define scope of configs
    
    n_comps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]   #no. of PCA, set the upper limit to 30
    #n_comps = [10]
     
    # create configs
    configs = list()
    
    for k in n_comps:
        cfg = k
        configs.append(cfg)

    print('Total configs: %d' % len(configs))
    return configs


# In[42]:


cfg_list = model_configs()


# In[43]:


cfg_list = model_configs()
scores = grid_search(cfg_list)
print('done')


# In[44]:


scores


# In[45]:


steps_final = [('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
               ('pca', PCA(n_components = 20, random_state=1)), #chosen PCA = 20
               ('ols', LinearRegression())
              ]

pipeline_final = Pipeline(steps_final)


# In[46]:


import time

start=time.time()    
   
n_train = 20

n_records = len(X)

y_test_list = []
    
y_pred_list = []

j = 0

for i in range(n_train, n_records):
    
    X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
        
    model = pipeline_final.fit(X_train, y_train)            

    y_pred = model.predict(X_test)
    
    y_pred_list.extend(y_pred)
    
    y_test_list.extend(y_test)

    j += 1
    
end=time.time()

print("Running Time:", end - start)


# In[47]:


pd.options.display.max_rows=350
result_PCA = pd.DataFrame({'Inflation_Actual': y[-n_forecast-5:], 'Inflation_Predicted': y_pred_list}, columns=['Inflation_Actual', 'Inflation_Predicted'])
result_PCA


# In[48]:


from sklearn.metrics import mean_squared_error
actual = result_PCA['Inflation_Actual']
predicted = result_PCA['Inflation_Predicted']
mse1 = mean_squared_error(actual, predicted)
print("Mean Squared Error:", mse1)


# In[49]:


fig, ax = plt.subplots(figsize=(20, 10)) 
result_PCA.plot(ax=ax)


# In[50]:


#KBest
model_KBest = SelectKBest(score_func=f_regression, k=10) #select 10 best features

model_KBest = model_KBest.fit(X, y)

KBest = model_KBest.get_support()
KBest


# In[51]:


#Identifying the selected features

KBest_feature = []

for i in range(0,KBest.shape[0]):
    if KBest[i] == True:
        K_feature = X.columns[i]
        KBest_feature.append(K_feature)
    
KBest_feature  


# In[52]:


n_train = 25   # 2013-M12
n_records = len(X)
n_forecast = n_records-n_train

j = 0
for i in range(n_train, n_records):    
    #train, test = X[0:i], X[i:i+1]   #expanding window
    train, test = X[j:i], X[i:i+1]   #rolling window
#    print('train=%d, test=%d' % (len(train), len(test)))
    j += 1


# In[53]:


# pipeline
def pipeline(config):

    # unpack config
    n_features = config

    # Steps
    steps = [('SelectKBest', SelectKBest(score_func=f_regression, k=n_features)),
             ('ols', LinearRegression())
            ]

    pipeline = Pipeline(steps)

    return pipeline


# In[54]:


def walk_forward_validation(cfg):
    
    n_train = 25

    n_records = len(X)

    y_test_list = []
    
    y_pred_list = []

    j = 0

    for i in range(n_train, n_records):
    
        X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
       
        model = pipeline(cfg).fit(X_train, y_train)        
    
        y_pred = model.predict(X_test)
    
        y_pred_list.extend(y_pred)
    
        y_test_list.extend(y_test)

        j += 1
    
    score_rmse = metrics.mean_squared_error(y_test_list, y_pred_list, squared=False)
    
    print(' > %.3f' % score_rmse)
    
    return score_rmse 


# In[55]:


# score a model, return None on failure
def repeat_evaluate(config, n_repeats=1):
    # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(config) for _ in range(n_repeats)]
    # summarize score
    result = np.mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# In[56]:


# grid search configs
def grid_search(cfg_list):
    # evaluate configs
    scores = [repeat_evaluate(cfg) for cfg in cfg_list]
    # sort configs by score_rmse, asc
    #scores.sort(key=lambda tup: tup[1])
    scores.sort(key=lambda tup: tup[1], reverse=True)
    return scores


# In[57]:


# create a list of configs to try
def model_configs():
    # define scope of configs
      
    n_features = [2,4,6,8,10,12,14,16,18,20,22,24,26,28]

    #n_features = [10]
     
    # create configs
    configs = list()
    
    for k in n_features:
        cfg = k
        configs.append(cfg)

    print('Total configs: %d' % len(configs))
    return configs


# In[58]:


cfg_list = model_configs()


# In[59]:


cfg_list = model_configs()
# grid search
scores = grid_search(cfg_list)
print('done')


# In[60]:


scores


# In[61]:


# Pipeline

# Steps
steps_final = [('SelectKBest', SelectKBest(score_func=f_regression, k=4)),
               ('ols', LinearRegression())
              ]

pipeline_final = Pipeline(steps_final)


# In[62]:


start=time.time()    
   
n_train = 25

n_records = len(X)

y_test_list = []
    
y_pred_list = []

j = 0

for i in range(n_train, n_records):
    
    X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
        
    model = pipeline_final.fit(X_train, y_train)            

    y_pred = model.predict(X_test)
    
    y_pred_list.extend(y_pred)
    
    y_test_list.extend(y_test)

    j += 1
    
end=time.time()

print("Running Time:", end - start)


# In[63]:


mse2 = metrics.mean_squared_error(y_test_list, y_pred_list, squared=False)
mse2


# In[64]:


pd.options.display.max_rows=300
result_KBest = pd.DataFrame({'Inflation_Actual': y[-n_forecast:], 'Inflation_Predicted': y_pred_list}, columns=['Inflation_Actual', 'Inflation_Predicted'])
result_KBest


# In[65]:


fig, ax = plt.subplots(figsize=(20, 10)) 
result_KBest.plot(ax=ax)


# In[66]:


#RFE

ols = LinearRegression()
model_rfe = RFE(ols, n_features_to_select=10)
model_rfe = model_rfe.fit(X, y)

#Select 10 features first


# In[67]:


rfe = model_rfe.support_
rfe


# In[68]:


rfe_features = []

for i in range(0,rfe.shape[0]):
    if rfe[i] == True:
        rfe_feature = X.columns[i]
        rfe_features.append(rfe_feature)
    
rfe_features


# In[69]:


n_train = 25   
n_records = len(X)
n_forecast = n_records-n_train

j = 0
for i in range(n_train, n_records):    
    #train, test = X[0:i], X[i:i+1]   #expanding window
    train, test = X[j:i], X[i:i+1]   #rolling window
#    print('train=%d, test=%d' % (len(train), len(test)))
    j += 1


# In[70]:


# pipeline
def pipeline(config):

    # unpack config
    n_features = config

    # Steps
    steps = [('rfe', RFE(LinearRegression(), n_features_to_select=n_features)),
             ('ols', LinearRegression())
            ]

    pipeline = Pipeline(steps)

    return pipeline


# In[71]:


def walk_forward_validation(cfg):
    
    n_train = 25

    n_records = len(X)

    y_test_list = []
    
    y_pred_list = []

    j = 0

    for i in range(n_train, n_records):
    
        X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
       
        model = pipeline(cfg).fit(X_train, y_train)        
    
        y_pred = model.predict(X_test)
    
        y_pred_list.extend(y_pred)
    
        y_test_list.extend(y_test)

        j += 1
    
    score_rmse = metrics.mean_squared_error(y_test_list, y_pred_list, squared=False)
    
    print(' > %.3f' % score_rmse)
    
    return score_rmse 


# In[72]:


# score a model, return None on failure
def repeat_evaluate(config, n_repeats=1):
    # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(config) for _ in range(n_repeats)]
    # summarize score
    result = np.mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# In[73]:


# grid search configs
def grid_search(cfg_list):
    # evaluate configs
    scores = [repeat_evaluate(cfg) for cfg in cfg_list]
    # sort configs by score_rmse, asc
    #scores.sort(key=lambda tup: tup[1])
    scores.sort(key=lambda tup: tup[1], reverse=True)
    return scores


# In[74]:


# create a list of configs to try
def model_configs():
    # define scope of configs
      
    n_features = [5,10,15,20,25]

    #n_features = [10]
     
    # create configs
    configs = list()
    
    for k in n_features:
        cfg = k
        configs.append(cfg)

    print('Total configs: %d' % len(configs))
    return configs


# In[75]:


# model configs
cfg_list = model_configs()


# In[76]:


cfg_list = model_configs()
# grid search
scores = grid_search(cfg_list)
print('done')


# In[77]:


scores


# In[78]:


# Pipeline

# Steps
steps_final = [('rfe', RFE(LinearRegression(), n_features_to_select=15)),
               ('ols', LinearRegression())
              ]

pipeline_final = Pipeline(steps_final)


# In[79]:


start=time.time()    
   
n_train = 25

n_records = len(X)

y_test_list = []
    
y_pred_list = []

j = 0

for i in range(n_train, n_records):
    
    X_train, X_test, y_train, y_test = X[j:i], X[i:i+1], y[j:i], y[i:i+1]
        
    model = pipeline_final.fit(X_train, y_train)            

    y_pred = model.predict(X_test)
    
    y_pred_list.extend(y_pred)
    
    y_test_list.extend(y_test)

    j += 1
    
end=time.time()

print("Running Time:", end - start)


# In[80]:


mse3 = metrics.mean_squared_error(y_test_list, y_pred_list, squared=False)
mse3


# In[81]:


pd.options.display.max_rows=300
result_RFE = pd.DataFrame({'Inflation_Actual': y[-n_forecast:], 'Inflation_Predicted': y_pred_list}, columns=['Inflation_Actual', 'Inflation_Predicted'])
result_RFE


# In[82]:


fig, ax = plt.subplots(figsize=(20, 10)) 
result_RFE.plot(ax=ax)


# In[83]:


result_PCA.plot()
result_KBest.plot()
result_RFE.plot()


# In[84]:


data = {'mean squared error': [mse1, mse2, mse3]}

index = ['PCA', 'K_best', 'RFE']

df = pd.DataFrame(data, index=index)
print(df)


# In[ ]:




