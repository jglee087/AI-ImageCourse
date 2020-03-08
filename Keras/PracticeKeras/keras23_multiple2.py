import numpy as np
from numpy import array

def split_sequence2(sequence, n_steps):
    X, y =list(), list()
    for i in range(len(sequence)):
        end_ix=i + n_steps
        if end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix-1,-1]
        X.append(seq_x)
        y.append(seq_y)        
    return array(X),array(y)

in_seq1 = np.array( [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = np.array( [15, 25, 35, 45, 55, 65, 75, 85, 95, 105])

out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# print(in_seq1.shape) # (10,)
# print(in_seq2.shape) # (10,)
# print(out_seq.shape) # (10,)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

# print(in_seq1.shape) # (10,1)
# print(in_seq2.shape) # (10,1)
# print(out_seq.shape) # (10,1)

dataset = np.hstack((in_seq1,in_seq2,out_seq))
# print(dataset.shape) # (10,3)

n_step = 3

x,y = split_sequence2(dataset,n_step)

#print(x.shape)
#print(y.shape)

# 실습
# 1. 함수 분석
# 2. LSTM 모델 구성 (지표는 loss)
#  [[ 90,  95],
#   [ 100, 105],
#   [110, 115]]])

from keras.models import Sequential
from keras.layers import Dense, LSTM


model=Sequential()
    
model.add(LSTM(8, input_shape=(3,2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))    
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=500, batch_size=1, verbose=0)

loss, mae = model.evaluate(x,y,batch_size=1)

x_pred = array([[ 90,  95],
   [ 100, 105],
   [110, 115]])

x_pred = x_pred.reshape(1,3,2)
y_pred = model.predict(x_pred,batch_size=1)
print(y_pred)
