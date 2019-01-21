import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.labels
test_labels = mnist.test.labels

print("train_images_shape:", train_images.shape)
print("train_labels_shape:", train_labels.shape)
print("test_images_shape:", test_images.shape)
print("test_labels_shape:", test_labels.shape)
print("train_images:", train_images[0])
print("train_images_length:", len(train_images[0]))
print("train_labels:", train_labels[0])

import numpy as np
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None,10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


pred = tf.nn.softmax(tf.matmul(x, W) + b)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), 1))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    beginTime = datetime.datetime.now()

    for epoch in range(25):  #0.92,33s； 60同样
        avg_cost = 0.
        iretation = int(mnist.train.num_examples/100)
        
        for i in range(iretation):
            batch_xs, batch_ys = mnist.train.next_batch(100)#此处next_batch是如何读入batch_xs
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cross_entropy, {x: batch_xs, y: batch_ys})/iretation
            
        if(epoch+1)%5 ==0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
          
    print("Done")
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    endTime = datetime.datetime.now()
    print("time used: {}".format(endTime - beginTime))
