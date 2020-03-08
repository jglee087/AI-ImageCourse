import numpy as np

from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])

x_test=np.array([11,12,13,14,15,16,17,18,19,20])
y_test=np.array([11,12,13,14,15,16,17,18,19,20])

# print(y.shape)

#2. 모델 구성
model=Sequential()

#model.add(Dense(16, input_shape=(1,)))
model.add(Dense(16, input_dim=1))
model.add(Dense(12))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=200, batch_size=1,verbose=0)

#4. 평가예측
loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print('MSE:',mse)

#5. 결과예측
x_pred=np.array([500,560,660])
res1=model.predict(x_pred, batch_size=1)
print('\nResult1:',res1)
