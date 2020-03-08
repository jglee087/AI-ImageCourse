import numpy as np
import pandas as pd

# index_col=0 첫 번째 열을 column으로 하겠다.
# header=0 첫 번째 행을 row으로 하겠다.

df1=pd.read_csv('./samsung.csv',encoding='cp949', index_col=0, header=0, sep=',', thousands=',')
df2=pd.read_csv('./kospi200.csv',encoding='cp949', index_col=0, header=0, sep=',', thousands=',')

df1=df1.sort_values(['일자'], ascending=True)
df2=df2.sort_values(['일자'], ascending=True)

# padnas를 numpy로 변환

df1 = df1.values
df2 = df2.values

np.save('./data/samsung.npy',arr=df1)
np.save('./data/kospi.npy',arr=df2)


import numpy as np
import pandas as pd

samsung = np.load('./data/samsung.npy')
kospi200 = np.load('./data/kospi.npy')

def split_xy5(dataset, time_steps, y_column):
    
    x,y =list(), list()
    
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset)+1:
            break

        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,3]

        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)

leng = 21

x, y = split_xy5(samsung, leng, 1)

prediction = x[-1,:]
x = x[:-1,:]
y = np.array(y[:-1])

# 데이터 셋 나누기

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=1, shuffle=True)
print(x_train.shape,x_test.shape)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb

xf_train=x_train.reshape(-1,leng*5)
#yf_train=y_train.reshape(-1,75)
xf_test=x_test.reshape(-1,leng*5)
#yf_test=y_test.reshape(-1,75)

prediction_f=prediction.reshape(-1,leng*5)

re=[]
va=[]

for i in range(5):
    import warnings
    warnings.filterwarnings(action='ignore') 

    mod = GradientBoostingRegressor()
    mod.fit(xf_train,y_train)
    res=mod.score(xf_test,y_test) # Accuracy만 반환
    val=mod.predict(prediction_f)

    re.append(res)
    va.append(val)
    
index=(np.argsort(re))
re=np.array(re)
va=np.array(va)

re=re[index]
va=va[index]

for i in range(5):
    print(re[i],va[i])
    
# 데이터 전처리

from sklearn.preprocessing import StandardScaler, MinMaxScaler

x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]  ))
x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]  ))

#scaler1=StandardScaler()
scaler1=MinMaxScaler()
scaler1.fit(x_train)

x_train_scaled = scaler1.transform(x_train)
x_test_scaled = scaler1.transform(x_test)

x_train_scaled = x_train_scaled.reshape(-1,leng,5)
x_test_scaled = x_test_scaled.reshape(-1,leng,5)

from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Activation
from keras.models import Sequential

from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam


model = Sequential()
model.add(LSTM(128,activation='tanh',input_shape=(leng,5)))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256,activation='tanh'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.002),metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)

hist=model.fit(x_train_scaled, y_train, epochs= 60, batch_size=4,verbose=2,validation_split=0.2,callbacks=[early_stopping])

loss, mae = model.evaluate(x_test_scaled,y_test,batch_size=4)
print('RMSE:',np.sqrt(loss),'MAE: ',mae)

y_pred=model.predict(x_test_scaled)

for i in range(5):
    print('Final: ', y_test[i], '/ 예측가:', y_pred[i])

from sklearn.metrics import r2_score
r2_sco=r2_score(y_test, y_pred)
print(r2_sco)

pred = prediction

pred = pred.reshape(1,leng*5)
pred_scaled=scaler1.transform(pred)
pred_scaled=pred_scaled.reshape(-1,leng,5)

y_predict = model.predict(pred_scaled,batch_size=4)
print(y_predict)

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

xx1=(np.sqrt(hist.history['loss']))
xx2=(np.sqrt(hist.history['val_loss']))

loss_ax.plot(xx1, 'y', label='train loss')
loss_ax.plot(xx2, 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

plt.show()

# 삼성전자 예측가격:

# 1. RandomForest: 60626(60,600)원
# 2. LSTM: 58485(58,500)원
# 21