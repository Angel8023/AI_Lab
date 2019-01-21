import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

def save_restore:
    @下面一段写在with Session之前，何处定义均可
    ckpt_dir = "D:\软件数据\VScode\ckpt_dir"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Call this after declaring all tf.Variables.
    saver = tf.train.Saver()
    # This variable won't be stored, since it is declared after tf.train.Saver()
    non_storable_variable = tf.Variable(777)


    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
    @下面一段跟在with Session里 初始化所有变量之后即可
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
                #(ckpt.model_checkpoint_path +'.meta')   # 载入图结构，保存在.meta文件中
        start = global_step.eval() # get last global_step
        print("Start from:", start)




        for i in range(start, 100):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_input: 0.8, p_keep_hidden: 0.5})
    @下面一段跟在sess.run(optimizer,loss)即训练之后
            global_step.assign(i).eval() # set and update(eval) global_step with index, i
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)