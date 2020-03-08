import numpy as np
from numpy import array

def split_sequence(sequence, n_steps):
    X, y =list(), list()
    for i in range(len(sequence)):
        end_ix=i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)        
    return array(X),array(y)

dataset=[10,20,30,40,50,60,70,80,90,100]
n_step = 3
x,y = split_sequence(dataset,n_step)


val_apply = False

dnn_app = True
lstm_app = False

if lstm_app == True:
    x = x.reshape(x.shape[0], x.shape[1], 1)
    
if val_apply == False:
    x_train, x_test = x[:6], x[6:]
    y_train, y_test = y[:6], y[6:]

else:
    x_train, x_val, x_test = x[:5], x[5:6], x[6:]
    y_train, y_val, y_test = y[:5], y[5:6], y[6:]
    
# for i in range(len(x)):
#     print(x[i],y[i])

from keras.models import Sequential
from keras.layers import Dense, LSTM

model=Sequential()
    
if dnn_app == True:

    model.add(Dense(8, input_shape=(3,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))    
    model.add(Dense(1))

if lstm_app == True:
    
    model.add(LSTM(16, input_shape=(3,1), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

if val_apply == False:
    model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=0)
else:
    model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=0,validation_data=[x_val,y_val])

loss, mae = model.evaluate(x_test,y_test,batch_size=1)
print('Loss: ',loss, 'MAE: ',mae)

x_pred = array([[90, 100, 110]]) # (3,)

if lstm_app == True:
    x_pred=x_pred.reshape(1,3,1)
    
y_pred = model.predict(x_pred,batch_size=1)

print(y_pred)