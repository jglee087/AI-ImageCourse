import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

#1. 데이터

x=np.array([range(1,101),range(101,201)])
y=np.array([range(1,101),range(101,201)])

# print(x.shape) # (2,100)
# print(y.shape) # (2,100)

x=np.transpose(x)
y=np.transpose(y)


# print(x.shape) # (100,2)
# print(y.shape) # (100,2)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,train_size=0.8,
                                                    shuffle=False,random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.25, train_size =0.75, shuffle=False,
                                                  random_state=66)

in_dim=x.shape[1]
out_dim=y.shape[1]


#2. 모델 구성
model=Sequential()

#model.add(Dense(10, input_shape=(2,)))

#model.add(Dense(32*2, input_dim=in_dim)) # input dimension
model.add(Dense(16, input_dim=2)) # input dimension
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2))
#model.add(Dense(out_dim)) # output dimension

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=10,verbose=0,validation_data=(x_val,y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test,y_test,batch_size=10)
print('RMSE:',np.sqrt(mse))


#5. 결과예측
xx=np.array([[201,202,203],[201,202,203] ])
x_pred=np.transpose(xx)

res1=model.predict(x_pred, batch_size=10)
print('\nResult1:',res1)

y_predict=model.predict(x_test,batch_size=10)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('\nRMSE: ', RMSE(y_test,y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y_test,y_predict)
print('R2: ',r2_y_predict)
