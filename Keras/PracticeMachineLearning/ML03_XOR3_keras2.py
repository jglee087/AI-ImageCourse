from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from keras.utils import to_categorical

#1. 데이터
x_train = [ [0,0], [1,0], [0,1], [1,1] ]
y_train = [ 0, 1, 1, 0]

x_train = np.array(x_train)
y_train = np.array(y_train)

# #2. 모델

model=Sequential()

model.add(Dense(8, activation='relu',input_shape=(2,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=2,epochs=50,verbose=0)

#4. 평가

x_test =np.array( [ [0,0], [1,0], [0,1], [1,1] ] )
y_predict = model.predict(x_test)

y_predict[y_predict>0.5]=1
y_predict[y_predict<0.5]=0

print(x_test,"의 예측 결과:",y_predict.astype(int))