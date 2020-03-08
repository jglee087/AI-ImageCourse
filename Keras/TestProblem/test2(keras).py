#!/usr/bin/env python
# coding: utf-8


import numpy as np

from keras import models, layers
from keras.utils.np_utils import to_categorical
from keras import optimizers


# In[35]:


#1. 데이터
x=np.array([[1,2,3,4,5,6,7,8,9,10]])
y=np.array([[1,0,1,0,1,0,1,0,1,0]])

x=np.transpose(x)
y=np.transpose(y)

# In[36]:


#2. 모델 구성

model=models.Sequential()

model.add(layers.Dense(64, input_dim=1))
model.add(layers.Dense(128))
model.add(layers.Dense(256))
model.add(layers.Dense(128))
model.add(layers.Dense(64))

model.add(layers.Dense(1,activation='sigmoid'))

# In[38]:


#3. 훈련

adam = optimizers.adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x,y, epochs=200, batch_size=1,verbose=0)


# In[39]:


#4. 평가예측
loss, accuracy = model.evaluate(x,y,batch_size=1)                            
print('Acc:',accuracy)


# In[40]:


#5. 결과예측
x_pred=np.array([11,12,13])
res1=model.predict(x_pred, batch_size=1)
res1=np.argmax(res1,1)
print('\nResult1:',res1)




