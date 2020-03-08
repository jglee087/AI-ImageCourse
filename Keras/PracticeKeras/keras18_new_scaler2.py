
import numpy as np

x = np.array(range(1,21))
y = np.array(range(1,21))

x = x.reshape(20,1)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( \
        x,y,test_size=0.5, random_state= 66, shuffle = False)

print("=============================")
print(x_train)
print("=============================")
print(x_test)
print("=============================")

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("=============================")
print(x_train)
print("=============================")
print(x_test)
print("=============================")
