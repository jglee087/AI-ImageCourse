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


x, y = split_xy5(samsung, 5, 1)
print(x.shape, y.shape)

# 데이터 셋 나누기

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=1, shuffle=False)
print(x_train.shape,x_test.shape)

# 데이터 전처리

from sklearn.preprocessing import StandardScaler

x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]  ))
x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]  ))
# x_train=x_train.reshape(-1,5,5)
# x_test=x_test.reshape(-1,5,5)

scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler1.transform(x_train)
x_test_scaled = scaler1.transform(x_test)

from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(25,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, epochs= 120, batch_size=1,verbose=0,validation_split=0.2,callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled,y_test,batch_size=1)
print('RMSE:',np.sqrt(mse))

y_pred=model.predict(x_test_scaled)

for i in range(5):
    print('Final: ', y_test[i], '/ 예측가:', y_pred[i])

pred = np.array( [ [61800, 61800, 60700, 60800, 14916555],
         [59400, 59400, 58300, 58800, 23664541],
         [59100, 59700, 58800, 59100, 16446102],
         [58800, 58800, 56800, 57200, 20821939],
         [57800, 58400, 56400, 56400, 19749457] ]) 

pred = pred.reshape(-1,25)
pred_scaled=scaler1.transform(pred)

y_predict = model.predict(pred_scaled,batch_size=2)
print(y_predict)
