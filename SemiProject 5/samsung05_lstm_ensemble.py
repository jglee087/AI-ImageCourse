import numpy as np
import pandas as pd

samsung = np.load('./data/samsung.npy')
kospi200 = np.load('./data/kospi.npy')

def split_xy5(dataset, time_steps, y_column):
    
    x,y =list(), list()
    
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break

        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,3]

        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)


x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)

#print(x.shape, y.shape)

# 데이터 셋 나누기

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, train_size=0.7, random_state=1, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, train_size=0.7, random_state=1, shuffle=False)


#print(x_train.shape,x_test.shape)

# 데이터 전처리

from sklearn.preprocessing import StandardScaler

x1_train = np.reshape( x1_train, (x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2]  ))
x1_test = np.reshape( x1_test, (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]  ))

x2_train = np.reshape( x2_train, (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]  ))
x2_test = np.reshape( x2_test, (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]  ))

scaler1=StandardScaler()
scaler1.fit(x1_train)

x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

x2_train_scaled = scaler1.transform(x2_train)
x2_test_scaled = scaler1.transform(x2_test)

x1_train_scaled = x1_train_scaled.reshape(-1,5,5)
x1_test_scaled = x1_test_scaled.reshape(-1,5,5)

x2_train_scaled = x2_train_scaled.reshape(-1,5,5)
x2_test_scaled = x2_test_scaled.reshape(-1,5,5)

from keras.layers import Dense, Input
from keras.models import Sequential, Model

input1 = Input(shape=(5,5))
dense1 = LSTM(64,activation='relu')(input1)
dense2 = Dense(32,activation='relu')(dense1)
dense3 = Dense(64,activation='relu')(dense2)
output1 = Dense(32,activation='relu')(dense3)

input2 = Input(shape=(5,5))
dense_1 = LSTM(64,activation='relu')(input2)
dense_2 = Dense(32,activation='relu')(dense_1)
dense_3 = Dense(64,activation='relu')(dense_2)
output2 = Dense(32,activation='relu')(dense_3)

from keras.layers import Concatenate
merge1 = Concatenate()([output1, output2])

middle1 = Dense(32,activation='relu')(merge1)
middle2 = Dense(64,activation='relu')(middle1)
middle3 = Dense(32,activation='relu')(middle2)
middle4 = Dense(64,activation='relu')(middle3)
middle5 = Dense(32,activation='relu')(middle4)
output = Dense(1)(middle5)


model = Model(inputs = [input1,input2], outputs = output) 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit([x1_train_scaled,x2_train_scaled], y1_train, epochs= 80, batch_size=1,verbose=0,validation_split=0.2,callbacks=[early_stopping])

#history=model.fit([x1_train_scaled,x2_train_scaled], y1_train, epochs= 10, batch_size=1,verbose=1,validation_split=0.2,callbacks=[early_stopping])


loss, mse = model.evaluate([x1_test_scaled,x2_test_scaled],y1_test,batch_size=1)
print('RMSE:',np.sqrt(mse))

y_pred=model.predict([x1_test_scaled,x2_test_scaled])

for i in range(5):
    print('Final: ', y1_test[i], '/ 예측가:', y_pred[i])

# pred1 = np.array( [ [61800, 61800, 60700, 60800, 14916555],
#          [59400, 59400, 58300, 58800, 23664541],
#          [59100, 59700, 58800, 59100, 16446102],
#          [58800, 58800, 56800, 57200, 20821939],
#          [57800, 58400, 56400, 56400, 19749457] ]) 

# pred1 = pred1.reshape(-1,25)
# pred1_scaled=scaler1.transform(pred1)
# pred2 = pred1.reshape(-1,25)
# pred2_scaled=scaler1.transform(pred2)

# # print(pred_scaled.shape)
# y_predict = model.predict([pred1_scaled,pred2_scaled],batch_size=2)
# print(y_predict)
