#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install catboost --user
# !pip install ipywidgets --user
# !jupyter nbextension enable --py widgetsnbextension --user


# In[2]:


from catboost import CatBoostClassifier, Pool, cv


# In[3]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# In[4]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgbm


# In[5]:


from keras.layers import Dense, Input, Dropout, Activation, BatchNormalization, GaussianNoise
from keras.models import Model, load_model
from keras import optimizers, callbacks
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,Callback, EarlyStopping
from keras import backend as K
from swa.keras import SWA # swa


# # 데이터 불러오기

# In[6]:


# 데이터 불러오기
train = pd.read_csv('/home/lab21/data/train.csv', index_col=0)
train_2 = pd.read_csv('/home/lab21/data/train_x_0.2_99.8.csv', index_col=0)

test = pd.read_csv('/home/lab21/data/test.csv', index_col=0)
sample_submission = pd.read_csv('/home/lab21/data/sample_submission.csv', index_col=0)

# Train 데이터의 타입을 Sample_submission에 대응하는 가변수 형태로 변환
column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train['type_num'] = train['type'].apply(lambda x : to_number(x, column_number))

# 모델에 적용할 데이터 셋 준비 
#x = train.drop(columns=['type', 'type_num'], axis=1)
y = train['type_num']

x = train_2.drop(columns=['fiberID'], axis=1)

test_x = test.drop(columns=['fiberID'],axis=1)

x_name=x.columns
col_name=x_name


# In[7]:


x=np.array(x)
y=np.array(y)
test_x=np.array(test_x)


# In[8]:


x=x.astype('float32')
test_x=test_x.astype('float32')


# # Train_Test_SpIit

# In[9]:


from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,y,                                                     train_size=0.8, shuffle=True ,random_state=42)


# # Scaler

# In[10]:


from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler1=RobustScaler()
scaler2=StandardScaler()

scaler2.fit(x_train)
x_train=scaler2.transform(x_train)
x_test=scaler2.transform(x_test)

test_x =scaler2.transform(test_x)


# # Variables

# In[11]:


import datetime
start=datetime.datetime.now()
print(start)


# In[12]:


categorical_features_indices=19


# In[13]:


# model = CatBoostClassifier(
#     custom_loss=['Accuracy'],
#     random_seed=42,
#     logging_level='Silent'
# )


# In[14]:


# model.fit(x_train, y_train, plot=True)


# In[ ]:


# params = {'depth': [8, 10, 12],
#           'learning_rate' : [0.1, 0.15],
#          'l2_leaf_reg': [2,4],
#          'iterations': [500,800,1000]}

params = {'depth': [8,10,12 ],
          'learning_rate' : [0.08, 0.12 ],\
         #'l2_leaf_reg': [2,4],
         'iterations': [500,800,1200]}

cb = CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="accuracy", cv = 3)
cb_model.fit(x_train, y_train, verbose=0, plot=True, thread_count=24)


# In[ ]:


print(grid_cv.best_params_)
print(grid_cv.best_score_)
best_params = grid_cv.best_params_


# In[ ]:



# best_params.update({'od_type': 'Iter','od_wait': 100})


# In[ ]:


cat_clf = CatBoostClassifier(**best_params, verbose=2, thread_count=24)
cat_clf.fit(x_train, y_train)
print("test acc: {:.4f}".format(cat_clf.score(x_test, y_test)))


# In[ ]:


pred=cat_clf.predict_proba(test_x)


# In[ ]:


end=datetime.datetime.now()
print("걸린 시간:", end-start)


# # Model Predict

# In[ ]:


# 제출 파일 생성
submission = pd.DataFrame(data=pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('/home/lab21/20200225/csv/submission_data_CatBoost_1.csv', index=True)

