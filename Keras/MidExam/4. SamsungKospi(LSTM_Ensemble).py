import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Activation
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df1=pd.read_csv('samsung.csv', encoding='CP949', thousands = ',' )
df2=pd.read_csv('kospi200.csv', encoding='CP949', thousands = ',' )

df_1 = df1.sort_values(["일자"], ascending=True)
df_2 = df2.sort_values(["일자"], ascending=True)

try:
    del(df_1['일자'])
    del(df_2['일자'])
except:
    pass

df1_temp=np.array(df_1)
df2_temp=np.array(df_2)

print(df1_temp.shape,df2_temp.shape)

n_len=df1_temp.shape[0]
n_steps=5

x1=[]
y1=[]
x2=[]

for i in range(n_len-n_steps):
    x1_te=[df1_temp[i:i+n_steps,:]]
    y1_te=[df1_temp[i+n_steps,3]]
    x2_te=[df2_temp[i:i+n_steps,:]]

    x1.append(x1_te)
    y1.append(y1_te)
    x2.append(x2_te)   

x1=np.array(x1)
y1=np.array(y1)
x2=np.array(x2)

x1=x1.reshape(n_len-n_steps,n_steps,5)
y1=y1.reshape(n_len-n_steps,)
x2=x2.reshape(n_len-n_steps,n_steps,5)

print(x1.shape, y1.shape, x2.shape)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.9, shuffle=False)
x2_train, x2_test = train_test_split(x2, train_size=0.9,  shuffle=False)

input1 = Input(shape=(5,5))
lstm1 = LSTM(32, activation='relu',return_sequences=True)(input1)
lstm2 = LSTM(32, activation='relu')(lstm1)
dense1= Dense(64, activation='relu')(lstm2)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
output1 = Dense(32, activation='relu')(dense3)

input2 = Input(shape=(5,5))
lstm_1 = LSTM(32, activation='relu',return_sequences=True)(input2)
lstm_2 = LSTM(32, activation='relu')(lstm_1)
dense_1 = Dense(64, activation='relu')(lstm_2)
dense_2 = Dense(128, activation='relu')(dense_1)
dense_3 = Dense(64, activation='relu')(dense_2)
output2 = Dense(32, activation='relu')(dense_3)

from keras.layers import Concatenate
merge1 = Concatenate()([output1, output2])

middle1 = Dense(64, activation='relu')(merge1)
middle2 = Dense(64, activation='relu')(middle1)
middle3 = Dense(128, activation='relu')(middle2)
middle4 = Dense(64, activation='relu')(middle3)
middle5 = Dense(32, activation='relu')(middle4)
middle6 = Dense(16, activation='relu')(middle5)
output = Dense(1)(middle6)

model = Model(inputs = [input1,input2], outputs = output) 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train], epochs= 250, batch_size=1,verbose=0,validation_split=1./9.)

loss, mse = model.evaluate([x1_test,x2_test],y1_test,batch_size=1)
print('RMSE:',np.sqrt(mse))


temp = x1.shape[0]

pred1 = [df1_temp[temp:temp+n_steps,:]]
pred2 = [df2_temp[temp:temp+n_steps,:]]

pred1=np.array(pred1)
pred2=np.array(pred2)

y_pred = model.predict([pred1,pred2],batch_size=1)
y_pred





