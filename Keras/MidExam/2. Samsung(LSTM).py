import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Activation
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


df1=pd.read_csv('samsung.csv', encoding='CP949')
df2=pd.read_csv('kospi200.csv', encoding='CP949')

df_1 = df1.sort_values(["일자"], ascending=True)

try:
    del(df_1['일자'])
except:
    pass

df_temp=np.array(df_1)

import re

for i in range(df1.shape[0]):
    for j in range(df1.shape[1]-1):
        temp=re.findall("\d+",df_temp[i,j])
        te="".join(temp)
        df_temp[i,j]=te

print(df_temp.shape)

n_len=df_temp.shape[0]
n_steps=5

print(n_len, n_steps)

x=[]
y=[]

for i in range(n_len-n_steps):
    x1=[df_temp[i:i+n_steps,:]]
    y1=[df_temp[i+n_steps,3]]

    x.append(x1)
    y.append(y1)

x=np.array(x)
y=np.array(y)

x=[]
y=[]

for i in range(n_len-n_steps):
    x1=[df_temp[i:i+n_steps,:]]
    y1=[df_temp[i+n_steps,3]]

    x.append(x1)
    y.append(y1)

x=np.array(x)
y=np.array(y)

x=x.reshape(n_len-n_steps,n_steps,5)
y=y.reshape(n_len-n_steps,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, shuffle=False)
print(x.shape, y.shape)

model=Sequential()

model.add(LSTM(32, activation='relu',input_shape=(5,5),return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=200, batch_size=1,verbose=0,validation_split=1./9.)

loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print('RMSE:',np.sqrt(mse))

temp = x.shape[0]

pred = [df_temp[temp:temp+n_steps,:]]
pred=np.array(pred)
pred=pred.reshape(1,5,5)

y_pred = model.predict(pred,batch_size=1)
y_pred