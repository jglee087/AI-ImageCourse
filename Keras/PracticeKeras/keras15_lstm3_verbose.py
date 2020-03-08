import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x=array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], \
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60] ])
y=array( [4,5,6,7,8,9,10,11,12,13,50,60,70] ) #### ?????

# print(x.shape)
# print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델 구성
model=Sequential()

model.add(LSTM(16, activation='relu', input_shape=(3,1))) 

# LSTM 에서 input_shape는 (None, 열, 자르는 개수)
# Input data에서 reshape를 해주어야 한다.

model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))

model.add(Dense(1))

#model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=50, batch_size=1, verbose=0)

#4. 평가 예측
loss, mae = model.evaluate(x,y,batch_size=1)
print('Loss:',loss,'MAE: ',mae)

#5. 값 예측
x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90], \
                 [100,110,120] ]) # 
x_input = x_input.reshape(4,3,1) # (1,3,1)

y_pred=model.predict(x_input,batch_size=1)
print(y_pred)