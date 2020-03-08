import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x=array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], \
           [9,10,11], [10,11,12], [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], \
           [100,200,300] ] )
y=array( [4,5,6,7,8,9,10,11,12,13,50000,60000,70000, 400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler

x2=x.copy()
scaler2 = StandardScaler()
scaler2 = MinMaxScaler()

scalerApply = False

#scaler2.fit(x2)
#x2=scaler2.transform(x2)

# train은 10개, 나머지는 test

x_train=x2[:10]
x_test=x2[10:]
y_train=y[:10]
y_test=y[10:]

if scalerApply == True:
    scaler2.fit(x_train)
    x_train=scaler2.transform(x_train)
    scaler2.fit(x_test)
    x_test=scaler2.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성

model=Sequential()

model.add(LSTM(64, activation='relu', input_shape=(3,1)))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1))

model.summary()

#3. 훈련

model.compile(loss='mse', optimizer='adagrad', metrics=['mse'])
model.fit(x_train,y_train, epochs=250, batch_size=1,verbose=0)

#4. 평가예측

loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print(loss, mse)

#5. 결과예측

x_pred=np.array([[250,260,270]])

if scalerApply == True:
    scaler2.fit(x_pred)
    x_pred=scaler2.transform(x_pred)
    
x_pred=x_pred.reshape(1,3,1)

res1=model.predict(x_pred, batch_size=1)
print('\nResult1:',res1)


y_predict=model.predict(x_test,batch_size=1)

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y_test,y_predict)
print('R2: ',r2_y_predict)
