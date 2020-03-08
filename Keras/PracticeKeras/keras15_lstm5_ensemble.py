import numpy as np
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

from sklearn.model_selection import train_test_split

#1. 데이터
x1 = array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], \
            [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60] ])

x2 = array( [ [10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], \
      #       [70,80,90], [80,90,100], \
            [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6] ])

y1=array( [4,5,6,7,8,9,10,11,12,13,50,60,70] ) #### ?????
#y2=array( [40,50,60,70,80,90,100,110,120,130,5,6,7] ) #### ?????
y2=array( [40,50,60,70,80,90,100,130,5,6,7] ) #### ?????

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = \
# train_test_split(x1, x2, y1, y2, test_size=0.2,train_size=0.8,shuffle=False,random_state=66)

# x1_train, x1_val, x2_train, x2_val, y1_train, y1_val, y2_train, y2_val = \
# train_test_split(x1_train, x2_train, y1_train, y2_train, test_size = 0.25, train_size =0.75, shuffle=False,random_state=66)


#2. 모델 구성
input1 = Input(shape=(3,1))
dense1 = LSTM(8, activation='relu')(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(6)(dense2)
output1 = Dense(4)(dense3)

input2 = Input(shape=(3,1))
dense21 = LSTM(8)(input2)
dense22 = Dense(6)(dense21)
output2 = Dense(4)(dense22)

from keras.layers.merge import Concatenate, Add, Multiply
merge1 = Concatenate()([output1, output2])
#merge1= Multiply()([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(10)(middle1)
output = Dense(4)(middle2)

output_1 = Dense(16)(output) # 1번째 아웃풋 모델
output_1 = Dense(32)(output_1)
output_1 = Dense(1)(output_1)

output_2 = Dense(64)(output) # 2번째 아웃풋 모델
output_2 = Dense(16)(output_2)
output_2 = Dense(1)(output_2)

model = Model(inputs = [input1,input2], outputs = [output_1, output_2])
model.summary()


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1,x2], [y1,y2], epochs=100, batch_size=1, verbose=0)#, \
         #callbacks=[early_stopping])

#4. 평가 예측
loss = model.evaluate([x1,x2],[y1,y2],batch_size=1)
print(loss)
#print('\nLoss:',loss,',MAE: ',mae)

#5. 값 예측
x1_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90], \
                 [100,110,120] ]) # 
x2_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90], \
                 [100,110,120] ]) # 

x1_input = x1_input.reshape(4,3,1) # (1,3,1)
x2_input = x2_input.reshape(4,3,1) # (1,3,1)

y_pred=model.predict([x1_input,x2_input],batch_size=1)
print(y_pred)