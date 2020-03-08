import numpy as np

from keras.models import Model
from keras.layers import Dense, Input

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

#2. 함수형 모델 구성 (Functional Model)

# 앞 레이어의 이름을 명시

# input1 = Input(shape=(3,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(2)(dense1)
# dense3 = Dense(3)(dense2)
# output1 = Dense(1)(dense3)

input1 = Input(shape=(3,))
x = Dense(5)(input1)
x = Dense(2)(x)
x = Dense(3)(x)
output1 = Dense(1)(x)

## x는 소멸 될까?  그렇지 않다.

model = Model(inputs = input1, outputs = output1) # 함수형 모델은 최하단에서 명시 해준다.
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=2,verbose=0,validation_data=(x_val,y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test,y_test,batch_size=2)
print('RMSE:',np.sqrt(mse))


#5. 결과예측
x_pred=np.array([[103,104,105],[203,204,205],[403,404,405] ])
x_pred=np.transpose(x_pred)

res1=model.predict(x_pred, batch_size=2)
print('\nResult1:',res1)

y_predict=model.predict(x_test,batch_size=2)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('\nRMSE: ', RMSE(y_test,y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y_test,y_predict)
print('R2: ',r2_y_predict)

