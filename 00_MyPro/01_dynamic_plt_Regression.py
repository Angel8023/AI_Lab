import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
x_data = np.float32(np.random.rand(100))
y_data = np.square(x_data) + 0.3

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W*x_data + b 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)#分成1×1占用第一个，如果212占用第二个即第二行
ax.scatter(x_data,y_data)#以点的形式显示
plt.ion()#show一次不暂停，连续show
plt.show()

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(0,2001):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(W),sess.run(b))
        try:
                ax.lines.remove(lines[0])#去除lines的第一个线段
        except Exception:
                pass
        predict_y = sess.run(y)
        lines = ax.plot(x_data,predict_y,'r-',lw=5)#以线的形式plot上去,红，粗细为5
        plt.pause(0.1)