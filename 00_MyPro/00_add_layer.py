import tensorflow as tf
import numpy as np 


def add_layer(inputs,in_size,out_size,activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))#tf中x属性排为一行,Weights仅起连接两个神经元作用[N0，N1]
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1 )#默认为Variable
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#数据处理部分
x_data = np.linspace(-1,1,300)[:,np.newaxis]#-1：1的等差数列，300行，加一维为列
noise = np.random.normal(0,0.5,x_data.shape)#均值，方差0.5，加入噪点使其不严格符合直线
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1],name='x_input')#第一个是数据个数，第二个是数据多少个属性
ys = tf.placeholder(tf.float32,[None,1],name='y_input')#多少个神经元输出

#定义model，输入层一个属性即一个神经元，隐藏层10个，输出层一个
l1 = add_layer(xs,1,10,layer_num=1,activation_function=tf.nn.relu)#输入到隐藏，出来是隐藏层的参数进下一层
l2 = add_layer(l1,10,20,layer_num=2,activation_function=tf.nn.relu)#不relu梯度爆炸
y = add_layer(l2,20,1,layer_num='out',activation_function=None)

#用Loss寻找最佳参数模型
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-ys),axis = 1))
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#开始训练部分
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(0,2001):
        sess.run(train,feed_dict = {xs:x_data,ys:y_data})
        if step%20 == 0:
            print(step,sess.run(loss,feed_dict = {xs:x_data,ys:y_data}))
       

