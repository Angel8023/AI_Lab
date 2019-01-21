import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)#第一次运行时会下载以后会直接打开

def add_layer(inputs,in_size,out_size,activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))#tf中x属性排为一行,Weights仅起连接两个神经元作用[N0，N1]
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1 )#默认为Variable
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})#要计算的形参传给算prediction所需的xs
    tf_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(tf_prediction,tf.float32))#把TrueFalse转为float值
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784],name='x_input')#第一个是数据个数，第二个是x有多少个属性
ys = tf.placeholder(tf.float32,[None,10],name='y_input')#第二个属性该层神经元数

#直接输入到输出，准确度1000次0.87
#l1 = add_layer(xs,784,625,activation_function=tf.nn.sigmoid)
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#用Loss寻找最佳参数模型
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis = 1))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#开始训练部分
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(0,2001):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
    if step%20 == 0:
        print(step,compute_accuracy(
            mnist.test.images, mnist.test.labels))#传入测试集的xs即test.images计算prediction

       

