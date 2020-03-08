import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

#1. 데이터
x=np.array(range(1,101))
y=np.array(range(1,101))

x=np.arange(1,101)
y=np.arange(1,101)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,train_size=0.8,
                                                    shuffle=False,random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.25, train_size =0.75, shuffle=False,
                                                  random_state=66)

#2. 모델 구성
model=Sequential()

#model.add(Dense(16, input_shape=(1,)))
model.add(Dense(64, input_dim=1))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['rmse'])
model.fit(x_train,y_train, epochs=200, batch_size=1,verbose=0,
          validation_data=(x_val,y_val))

#4. 평가예측
loss, rmse = model.evaluate(x_test,y_test,batch_size=1)
print('RMSE:',rmse)

#5. 결과예측
x_pred=np.array([1201,1202,1203])
res1=model.predict(x_pred, batch_size=1)
print('\nResult1:',res1)


    