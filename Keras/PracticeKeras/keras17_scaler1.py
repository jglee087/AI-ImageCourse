import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x=array( [ [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], \
           [9,10,11], [10,11,12], [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], \
           [100,200,300] ] )
y=array( [4,5,6,7,8,9,10,11,12,13,50000,60000,70000, 400], [~] )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# x1=x.copy()
# scaler1 = MinMaxScaler()
# scaler1.fit(x1)
# x1=scaler1.transform(x1)

x2=x.copy()
scaler2 = StandardScaler()
scaler2.fit(x2)
x2=scaler2.transform(x2)

# scaler3 = MaxAbsScaler()
# x3 = x.copy()
# scaler3.fit(x3)
# x3=scaler3.transform(x3)

# scaler4 = RobustScaler()
# x4 = x.copy()
# scaler4.fit(x4)
# x4=scaler4.transform(x4)

# ######### 

# xt=x.copy()
# scaler2.fit(xt)
# xt=scaler2.transform(xt)
# scaler1.fit(xt)
# xt=scaler1.transform(xt)

# train은 10개, 나머지는 test

#1. 데이터

x_train=x2[:10]
x_test=x2[10:]
y_train=y[:10]
y_test=y[10:]

#2. 모델 구성

model=Sequential()

model.add(Dense(32, input_shape=(3,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

model.summary()

#3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=50, batch_size=1,verbose=1)

#4. 평가예측

loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print(loss, mse)

#5. 결과예측

x_pred=np.array([[250,260,270]])
scaler2.fit(x_pred)
x_pred=scaler2.transform(x_pred)

res1=model.predict(x_pred, batch_size=1)
print('\nResult1:',res1)

y_predict=model.predict(x_test,batch_size=1)

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y_test,y_predict)
print('R2: ',r2_y_predict)
