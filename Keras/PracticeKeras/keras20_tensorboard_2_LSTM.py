import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

#1. 데이터
x=array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], \
           [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60] ])
y=array( [4,5,6,7,8,9,10,11,12,13,50,60,70] )

# print(x.shape)
# print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델 구성
model=Sequential()

## 1

# model.add(LSTM(10, activation='elu', input_shape=(3,1), return_sequences = True))
# model.add(LSTM(8, activation='elu', input_shape=(3,1), return_sequences = False))
# model.add(Dense(5, activation='elu'))
# model.add(Dense(1))

## 2

model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences = True))
model.add(LSTM(2, activation='elu', return_sequences = True))
model.add(LSTM(3, activation='tanh', return_sequences = True))
model.add(LSTM(5, activation='sigmoid', return_sequences = True))
model.add(LSTM(10, activation='exponential', return_sequences = False))
model.add(Dense(5, activation='elu'))
model.add(Dense(1))

## 3

# model.add(LSTM(10, activation='elu', input_shape=(3,1),return_sequences=True))
# #model.add(Reshape((1,10))) # (None,10) -> (None,10,1)
# model.add(LSTM(15, activation='elu')) #input_shape=(10,1)
# model.add(Dense(5, activation='elu'))
# model.add(Dense(1))

## 4

#model.add(LSTM(12, activation='elu', input_shape=(3,1),return_sequences=True))
# model.add(Dense(16, activation='elu'))
# model.add(LSTM(16, activation='elu',return_sequences=True))
# model.add(Dense(32, activation='elu'))
# model.add(LSTM(20, activation='elu',return_sequences=False))
# model.add(Dense(64, activation='elu'))
# model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

tb_hist = TensorBoard(log_dir="./graph", histogram_freq=0, \
                     write_graph=True, write_images=True)
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, verbose=2, \
         callbacks=[early_stopping, tb_hist])

#4. 평가 예측
loss, mae = model.evaluate(x,y,batch_size=1)
print('\nLoss:',loss,',MAE: ',mae)

#5. 값 예측
x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90], \
                 [100,110,120] ]) # 
x_input = x_input.reshape(4,3,1) # (1,3,1)

y_pred=model.predict(x_input,batch_size=1)
print(y_pred)