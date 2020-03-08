import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


#1. 데이터
x=array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7] ])
y=array( [4,5,6,7,8] ) #### ?????

# print(x.shape)
# print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(5,3,1)

#2. 모델 구성
model=Sequential()

model.add(LSTM(16, activation='relu', input_shape=(3,1))) 

# LSTM 에서 input_shape는 (None, 열, 자르는 개수)
# Input data에서 reshape를 해주어야 한다.


model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=150, batch_size=1, verbose=0)


#4. 평가예측

loss, mae = model.evaluate(x,y,batch_size=1)
print('MAE:',loss, mae)

#5. 

x_input = array([6,7,8]) # (3,)
x_input = x_input.reshape(1,3,1)

y_pred=model.predict(x_input,batch_size=1)
print(y_pred)
