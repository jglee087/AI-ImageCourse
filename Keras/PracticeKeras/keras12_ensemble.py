import numpy as np

from keras.models import Model
from keras.layers import Dense, Input

from sklearn.model_selection import train_test_split

#1. 데이터

x1 = np.array([range(1,101),range(101,201),range(301,401)])
x2 = np.array([range(1001,1101),range(1101,1201),range(1301,1401)])

y1 = np.array([range(1,101)])

# y2 = np.array([range(1101,1201)])

x1=np.transpose(x1) # (100,3)
x2=np.transpose(x2) # (100,3)
y1=np.transpose(y1) # (100,1) 

# y2=np.transpose(y2) # (100,1) 

#### Train_test_split 1

x1_train, x1_test = train_test_split(x1, test_size=0.2,train_size=0.8, shuffle=False,random_state=66)
x1_train, x1_val = train_test_split(x1_train, test_size = 0.25, train_size =0.75, shuffle=False, random_state=66)

x2_train, x2_test = train_test_split(x2, test_size=0.2,train_size=0.8, shuffle=False,random_state=66)
x2_train, x2_val = train_test_split(x2_train, test_size = 0.25, train_size =0.75, shuffle=False, random_state=66)

y1_train, y1_test = train_test_split(y1, test_size=0.2,train_size=0.8, shuffle=False,random_state=66)
y1_train, y1_val = train_test_split(y1_train, test_size = 0.25, train_size =0.75, shuffle=False, random_state=66)

#### Train_test_split 2

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, test_size=0.2,train_size=0.8,shuffle=False,random_state=66)

x1_train, x1_val, x2_train, x2_val, y1_train, y1_val = train_test_split(x1_train, x2_train, y1_train, test_size = 0.25, train_size =0.75, shuffle=False,random_state=66)

#2. 함수형 모델 구성 (Functional Model)
# 앞 레이어의 이름을 반드시 명시해줘야 한다.

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(5)(dense22)

# output1과 output2는 이전에 했던 것과 같이 꼭 1개여야 하는 것은 아니다. 왜냐하면
# y에 도착하기 전에는 모두 hidden layer층이기 때문에 input만 적절하면 
# concatenate하기 전에는 모두가 hidden layer이기 때문에 어떤 값을 가져도 된다.

# 소문자 concatenate
# from keras.layers.merge import concatenate # 모델을 합치는 방법
# merge1 = concatenate([output1, output2]) # 파라미터 두 개 이상은 리스트로 받는다.

# 대문자 Concatenate
from keras.layers import Concatenate
merge1 = Concatenate()([output1, output2])

middle1 = Dense(5)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

# 함수형 모델 2개를 concatenate한 ensemble model로 총 함수형 모델은 3개

model = Model(inputs = [input1,input2], outputs = output) # 함수형 모델은 최하단에서 명시 해준다.
# 입력이 여러개이면 리스트로 입력한다.

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train], epochs=250, batch_size=1,verbose=0,validation_data=([x1_val,x2_val],y1_val))

#4. 평가예측
loss, mse = model.evaluate([x1_test,x2_test],[y1_test],batch_size=1)
print('RMSE:',np.sqrt(mse))

#5. 결과예측
x1_pred=np.array([ [103,104,105],[203,204,205],[403,404,405] ])
x2_pred=np.array([ [1103,1104,1105],[1203,1204,1205],[1403,1404,1405] ])

x1_pred=np.transpose(x1_pred)
x2_pred=np.transpose(x2_pred)

res1=model.predict([x1_pred,x2_pred], batch_size=1)
print('\nResult1:',res1)

y_predict=model.predict([x1_test,x2_test],batch_size=1)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('\nRMSE: ', RMSE(y1_test,y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict=r2_score(y1_test,y_predict)
print('R2: ',r2_y_predict)