import numpy as np

from keras.models import Model
from keras.layers import Dense, Input

from sklearn.model_selection import train_test_split

#1. 데이터

x1 = np.array([range(1,101),range(101,201),range(301,401)])

# y1 = np.array([range(101,201)])

y1 = np.array([range(1,101),range(101,201),range(301,401)])
y2 = np.array([range(1001,1101),range(1101,1201),range(1301,1401)])
y3 = np.array([range(1,101),range(101,201),range(301,401)])

x1=np.transpose(x1) # (100,3)

y1=np.transpose(y1)
y2=np.transpose(y2)
y3=np.transpose(y3)

#### Train_test_split 1

x1_train, x1_test = train_test_split(x1, test_size=0.2,train_size=0.8, \
                                     shuffle=False,random_state=66)
x1_train, x1_val = train_test_split(x1_train, test_size = 0.25, train_size =0.75, \
                                    shuffle=False, random_state=66)

y1_train, y1_test = train_test_split(y1, test_size=0.2,train_size=0.8, \
                                     shuffle=False,random_state=66)
y1_train, y1_val = train_test_split(y1_train, test_size = 0.25, train_size =0.75, \
                                    shuffle=False, random_state=66)

y2_train, y2_test = train_test_split(y2, test_size=0.2,train_size=0.8, \
                                     shuffle=False,random_state=66)
y2_train, y2_val = train_test_split(y2_train, test_size = 0.25, train_size =0.75, \
                                    shuffle=False, random_state=66)

y3_train, y3_test = train_test_split(y3, test_size=0.2,train_size=0.8, \
                                     shuffle=False,random_state=66)
y3_train, y3_val = train_test_split(y3_train, test_size = 0.25, train_size =0.75, \
                                    shuffle=False, random_state=66)

#### Train_test_split 2

#x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = \
#train_test_split(x1, y1, y2, y3, test_size=0.2,train_size=0.8,shuffle=False,random_state=66)

#x1_train, x1_val, y1_train, y1_val, y2_train, y2_val,y3_train, y3_val = \
#train_test_split(x1_train, y1_train, y2_train, y3_train, test_size = 0.25, train_size =0.75, shuffle=False,random_state=66)

#2. 함수형 모델 구성 (Functional Model)
# 앞 레이어의 이름을 반드시 명시해줘야 한다.

input1 = Input(shape=(3,))
dense1 = Dense(64)(input1)
dense2 = Dense(128)(dense1)
dense3 = Dense(128)(dense2)
output1 = Dense(64)(dense3)

output_1 = Dense(64)(output1) # 1번째 아웃풋 모델
output_1 = Dense(64)(output_1)
output_1 = Dense(3)(output_1)

output_2 = Dense(64)(output1) # 2번째 아웃풋 모델
output_2 = Dense(128)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(128)(output1) # 3번째 아웃풋 모델
output_3 = Dense(64)(output_3)
output_3 = Dense(3)(output_3)

model = Model(inputs = input1, outputs = [output_1, output_2, output_3])
# 함수형 모델은 최하단에서 명시 해준다.
# 입력이 여러개이면 리스트로 입력한다.

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train],[y1_train,y2_train,y3_train], epochs=50, batch_size=1,verbose=0,validation_data=([x1_val],[y1_val,y2_val,y3_val]))

# loss => mse, mae
# metrics => mse, mae, accuracy

# #4. 평가예측
# loss, loss1, loss2, loss3, #1, #2, #3 = model.evaluate([x1_test,x2_test],[y1_test, y2_test, y3_test],batch_size=1)
# 첫번째 모델에 대한 loss는 loss1, #1는 #1
# 두번째 모델에 대한 loss는 loss2, #2는 #2
# 세번째 모델에 대한 loss는 loss3, #3는 #3 
# print('p:',loss,loss1,loss2,loss3,mse1,mse2,mse3)

loss = model.evaluate([x1_test],[y1_test, y2_test, y3_test],batch_size=1)
print('p:',loss)

# # #5. 결과예측
x1_pred=np.array([ [201,202,203],[204,205,206],[207,208,209] ])

x1_pred=np.transpose(x1_pred)

res1=model.predict([x1_pred], batch_size=1)
print('\nResult1:',res1)

y1_predict=model.predict([x1_test],batch_size=1)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_tes,y_pre):
    return np.sqrt(mean_squared_error(y_tes,y_pre))

rmse1 = RMSE(y1_test,y1_predict[0])
rmse2 = RMSE(y2_test,y1_predict[1])
rmse3 = RMSE(y3_test,y1_predict[2])

rmse = (rmse1+rmse2+rmse3)/3.

print('\nRMSE1: ',rmse1)
print('RMSE2: ',rmse2)
print('RMSE3: ',rmse3)
print('RMSE: ',rmse)

def RMSE2(y1_tes, y2_tes, y3_tes, y_pred):
    res1=np.sqrt(mean_squared_error(y1_tes,y_pred[0]))
    res2=np.sqrt(mean_squared_error(y2_tes,y_pred[1]))
    res3=np.sqrt(mean_squared_error(y3_tes,y_pred[2]))
    
    return res1, res2, res3

rmse_ = RMSE2(y1_test,y2_test,y3_test,y1_predict)
print('RMSE2: ',rmse_)


def RMSE3(y_tes, y_pred):
    res=[]
    s=len(y_pred)
    for i in range(s):    
        res.append(np.sqrt(mean_squared_error(y_tes[i],y_pred[i])))
    return res

rmsse=RMSE3([y1_test,y2_test,y3_test],y1_predict)
print('RMSE3: ',rmsse)

# R2 구하기
from sklearn.metrics import r2_score

r2_y1_pred=r2_score(y1_test,y1_predict[0])
r2_y2_pred=r2_score(y2_test,y1_predict[1])
r2_y3_pred=r2_score(y3_test,y1_predict[2])

r2_y_pred=(r2_y1_pred+r2_y2_pred+r2_y3_pred)/3.

print('\nR2_1: ',r2_y1_pred)
print('R2_2: ',r2_y2_pred)
print('R2_3: ',r2_y3_pred)
print('R2: ', r2_y_pred)