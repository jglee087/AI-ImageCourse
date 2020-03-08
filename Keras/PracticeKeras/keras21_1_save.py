import numpy as np
from time import time

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

#2. 모델 구성
model=Sequential()

#model.add(Dense(10, input_shape=(2,))) # input dimension
model.add(Dense(5, input_dim=3)) # input dimension
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1)) # output dimension

model.summary()

model.save('./save/savetest01.h5')
print("저장")
