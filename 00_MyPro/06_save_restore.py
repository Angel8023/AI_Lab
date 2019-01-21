import tensorflow as tf 

# W = tf.Variable([[1,2,3],[6,6,6]],dtype=tf.float32)
# b = tf.Variable([2,3,2],dtype=tf.float32)
# saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     save_path = saver.save(sess,'ckpt/w_b_.ckpt')
#     print('save to:',save_path)

W = tf.Variable(tf.zeros([2,3]))
b = tf.Variable(tf.zeros([3]))
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'ckpt/w_b_.ckpt')
    print(sess.run(W))