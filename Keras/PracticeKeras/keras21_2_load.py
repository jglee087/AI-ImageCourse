import numpy as np
from time import time

from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

from sklearn.model_selection import train_test_split

#1. 데이터

x=np.array([range(1,101),range(101,201),range(301,401)])
y=np.array([range(1,101)])

print(x.shape)
print(y.shape)

x=np.transpose(x) # (100,3)
y=np.transpose(y) # (100,1) 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,train_size=0.8,
                                                    shuffle=False,random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.25, train_size =0.75, shuffle=False,
                                                  random_state=66)

# 모델 불러오기

from keras.models import load_model

model=load_model('./save/savetest01.h5')

model.add(Reshape((-1,1)))
model.add(LSTM(10,name='LSTM'))
model.add(Dense(5,name='Dense1'))
model.add(Dense(6,name='Dense2'))
model.add(Dense(1,name='Dense3'))

model.summary()
#

from keras.callbacks import EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

tb_hist = TensorBoard(log_dir="./graph", histogram_freq=0, \
                     write_graph=True, write_images=True)


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=120, batch_size=100,verbose=0,validation_data=(x_val,y_val), \
         callbacks=[early_stopping, tb_hist])

#4. 평가예측
loss, mse = model.evaluate(x_test,y_test,batch_size=100)
print('RMSE:',np.sqrt(mse))


#5. 결과예측
x_pred=np.array([[103,104,105],[203,204,205],[403,404,405] ])
x_pred=np.transpose(x_pred)

res1=model.predict(x_pred, batch_size=100)
print('\nResult1:',res1)

y_predict=model.predict(x_test,batch_size=100)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('\nRMSE: ', RMSE(y_test,y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y_test,y_predict)
print('R2: ',r2_y_predict)

