import tensorflow as tf
import numpy as np
x_data = np.float32(np.random.rand(3, 100))#ax1+bx2+cx3+d
y_data = np.dot([0.500, 0.600,2.0], x_data) + 0.300  # y = w.t*x + b

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
y = tf.matmul(w, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b),sess.run(loss))
    