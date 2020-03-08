import numpy as np

def split_sequence2(sequence, n_steps):
    X, y =list(), list()
    for i in range(len(sequence)):
        end_ix=i + n_steps
        if end_ix > len(sequence)-2:
            break
        seq_x, seq_y = sequence[i:end_ix,:], sequence[end_ix+1,3]
        X.append(seq_x)
        y.append(seq_y)        
    return np.array(X),np.array(y)

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

for i in range(df_temp.shape[0]):
    for j in range(df_temp.shape[1]):
        temp=re.findall("\d+",df_temp[i,j])
        te="".join(temp)
        df_temp[i,j]=te

print(df_temp.shape)

n_len=df_temp.shape[0]
n_steps=6

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

x=x.reshape(n_len-n_steps,n_steps,5)
y=y.reshape(n_len-n_steps,)

print(x.shape, y.shape)

x=x.reshape(n_len-n_steps,n_steps*5)
y=y


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, shuffle=False)

model=Sequential()

model.add(Dense(64,input_shape=(n_steps*5,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=150, batch_size=1,verbose=0,validation_split=1./9.)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=150, batch_size=1,verbose=0,validation_split=1./9.)

temp = x.shape[0]

pred = [df_temp[temp:temp+n_steps,:]]
pred=np.array(pred)
pred=pred.reshape(1,n_steps*5)

y_pred = model.predict(pred,batch_size=1)
y_pred
