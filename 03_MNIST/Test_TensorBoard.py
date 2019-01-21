import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")

tf.summary.histogram("a", a)
tf.summary.scalar("b", b)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/test_tb', sess.graph).add_summary(sess.run(merged))
    sess.run(x)
#writer.close()