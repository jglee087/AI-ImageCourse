#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from keras import models, layers
from keras.utils.np_utils import to_categorical


# In[3]:


#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,1,2,3,4,5])

x=np.transpose(x)
y=np.transpose(y)

y=to_categorical(y) # one-hot vector
num_class=y.shape[1]


# In[4]:


#2. 모델 구성

model=models.Sequential()

model.add(layers.Dense(64,activation='relu', input_dim=1))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(num_class,activation='softmax'))


# In[5]:


#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=500, batch_size=1,verbose=0)


# In[6]:


#4. 평가예측
#loss, mae = model.evaluate(x,y,batch_size=1)
loss, accuracy = model.evaluate(x,y,batch_size=1)                            
#print('MAE:',mae)
print('Acc:',accuracy)


# In[7]:


#5. 결과예측
x_pred=np.array([11,12,13])
res1=model.predict(x_pred, batch_size=1)
res1=np.argmax(res1,axis=1)
print('\nResult1:',res1)




