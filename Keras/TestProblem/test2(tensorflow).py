#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np

x_data=[ [1],[2],[3],[4],[5],[6],[7],[8],[9],[10] ]

y_data=[ [1,0], [0,1], [1,0], [0,1], [1,0],[0,1], [1,0], [0,1], [1,0], [0,1] ]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

n1=64*2
n2=64*2
n3=64*2
n4=64*2

W1 = tf.Variable(tf.random_uniform([1, n1], -1., 1.))
W2 = tf.Variable(tf.random_uniform([n1, n2], -1., 1.))
W3 = tf.Variable(tf.random_uniform([n2, n3], -1., 1.))
W4 = tf.Variable(tf.random_uniform([n3, n4], -1., 1.))
W5 = tf.Variable(tf.random_uniform([n4, 2], -1., 1.))

# b1 = tf.Variable(tf.zeros([n1]))
# b2 = tf.Variable(tf.zeros([n2]))
# b3 = tf.Variable(tf.zeros([n3]))
# b4 = tf.Variable(tf.zeros([n4]))
# b5 = tf.Variable(tf.zeros([5]))

b1 = tf.Variable(tf.random_uniform([n1], -1., 1.))
b2 = tf.Variable(tf.random_uniform([n2], -1., 1.))
b3 = tf.Variable(tf.random_uniform([n3], -1., 1.))
b4 = tf.Variable(tf.random_uniform([n4], -1., 1.))
b5 = tf.Variable(tf.random_uniform([2], -1., 1.))
    
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2) 

L3 = tf.add(tf.matmul(L2, W3), b3)
L3 = tf.nn.relu(L3) 

L4 = tf.add(tf.matmul(L3, W4), b4)
L4 = tf.nn.relu(L4) 

model = tf.add(tf.matmul(L4, W5), b5)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(20000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 2000 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

print('\n\n')
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

print('\nPredict:',sess.run(prediction,feed_dict={X: [[11],[12],[13]]}))


# In[ ]:




