import numpy as np
#import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

# #1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델 구성

model=Sequential()

model.add(Dense(20, input_dim=1))
model.add(Dense(20))
model.add(Dense(1))

# #3. 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=200, batch_size=1,verbose=0)

#4. 평가예측
#loss, mae = model.evaluate(x,y,batch_size=1)
loss, mse = model.evaluate(x,y,batch_size=1)
#print('MAE:',mae)
print('MSE:',mse)

#5. 결과예측
x_pred=np.array([11,12,13])
res1=model.predict(x_pred, batch_size=1)
print('\nResult1:',res1)

res2=model.predict(x, batch_size=1)
print('\nResult2:',res2)