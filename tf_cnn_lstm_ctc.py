# tf CNN+LSTM+CTC 训练识别不定长数字字符图片
from freeTypeGenerateTextImage import GenerateCharListImage
import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 超参数
# 要生成的图片的大小
char_list_image_shape = (40, 120)
# 最大训练轮次
num_epochs = 10000
# 隐藏层神经元数量
num_hidden = 64
# 隐藏层层数
num_layers = 1
# 初始学习率
lr_start = 1e-3
# 学习率衰减因子
lr_decay_factor = 0.9
# 也可以用动量的优化算法
# momentum = 0.9
BATCHES = 10
train_batch_size = 64
test_batch_size = 64
TRAIN_SIZE = BATCHES * train_batch_size
# 训练迭代次数
iteration = 5000
# 每经过report_step次进行一些print操作，来输出当前的参数如loss值
report_step = 100
DIGITS = '0123456789'

obj_number = GenerateCharListImage()
# 类别为10位数字+blank+ctc blank
num_classes = obj_number.len + 1 + 1


# 生成batch_size个样本，样本的shape变为[batch_size,image_shape[1],image_shape[0]]
# 输入的图片是把每一行的数据看成一个时间间隔t内输入的数据，然后有多少行就是有多少个时间间隔
def get_next_batch(bt_size, img_shape):
	obj_batch = GenerateCharListImage()
	bt_x_inputs = np.zeros([bt_size, char_list_image_shape[1], char_list_image_shape[0]])
	bt_y_inputs = []
	for i in range(bt_size):
		# 生成不定长度的字符串及其对应的彩色图片
		color_image, text, text_vector = obj_batch.generate_color_image(img_shape, noise="gaussian")
		# 图片降噪，然后由彩色图片生成灰度图片的一维数组形式
		color_image = obj_batch.image_reduce_noise(color_image)
		gray_image_array = obj_batch.color_image_to_gray_image(color_image)
		# np.transpose函数将得到的图片矩阵转置成(image_shape[1]，image_shape[0])形状的矩阵，且由行有序变成列有序
		# 然后将这个图片的数据写入bt_x_inputs中第0个维度上的第i个元素
		bt_x_inputs[i, :] = np.transpose(gray_image_array.reshape((char_list_image_shape[0], char_list_image_shape[1])))
		# 把每个图片的标签添加到bt_y_inputs列表
		bt_y_inputs.append(list(text))
	# 将bt_y_inputs中的每个元素都转化成np数组
	targets = [np.asarray(i) for i in bt_y_inputs]
	# 将targets列表转化为稀疏矩阵
	sparse_matrix_targets = sparse_tuple_from(targets)
	# bt_size个1乘以char_list_image_shape[1]，也就是batch_size个样本中每个样本（每个样本即图片）的长度上的像素点个数（或者说列数）
	# seq_length就是每个样本中有多少个时间序列
	seq_length = np.ones(bt_x_inputs.shape[0]) * char_list_image_shape[1]
	# 得到的bt_x_inputs的shape=[bt_size, char_list_image_shape[1], char_list_image_shape[0]]
	return bt_x_inputs, sparse_matrix_targets, seq_length


# 解码稀疏矩阵
def decode_sparse_tensor(sparse_tensor):
	decoded_indexes = list()
	current_i = 0
	current_seq = []
	for offset, i_and_index in enumerate(sparse_tensor[0]):
		i = i_and_index[0]
		if i != current_i:
			decoded_indexes.append(current_seq)
			current_i = i
			current_seq = list()
		current_seq.append(offset)
	decoded_indexes.append(current_seq)
	result = []
	for index in decoded_indexes:
		result.append(decode_a_seq(index, sparse_tensor))
	return result


def decode_a_seq(indexes, spars_tensor):
	decoded = []
	for m in indexes:+
		str = DIGITS[spars_tensor[1][m]]
		decoded.append(str)
	return decoded


def report_accuracy(decoded_list, test_targets):
	original_list = decode_sparse_tensor(test_targets)
	detected_list = decode_sparse_tensor(decoded_list)
	true_numer = 0

	if len(original_list) != len(detected_list):
		print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
			  " test and detect length desn't match")
		return
	print("T/F: original(length) <-------> detectcted(length)")
	for idx, number in enumerate(original_list):
		detect_number = detected_list[idx]
		hit = (number == detect_number)
		print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
		if hit:
			true_numer = true_numer + 1
	acc = true_numer * 1.0 / len(original_list)
	print("Test Accuracy:", acc)


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
	"""
	:param sequences: 一个元素是列表的列表
	:param dtype: 列表元素的数据类型
	:return: 返回一个元组(indices, values, shape)
	"""
	indices = []
	values = []

	for index, seq in enumerate(sequences):
		# 每次取list中的一个元素，即一个list列表（代表的是一个图片的标签）
		# extend()函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
		# zip()函数将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的一个对象。
		# zip(a,b)函数分别从a和b中取一个元素组成元组，再次将组成的元组组合成一个新的迭代器。a与b的维数相同时，正常组合对应位置的元素。
		indices.extend(zip([index] * len(seq), range(len(seq))))
		values.extend(seq)

	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

	return indices, values, shape


# indices:二维int64的矩阵，代表非0的坐标点
# values:二维tensor，代表indice位置的数据值
# dense_shape:一维，代表稀疏矩阵的大小
# 假设sequences有2个，值分别为[1 3 4 9 2]、[ 8 5 7 2]。(即batch_size=2）
# 则其indices=[[0 0][0 1][0 2][0 3][0 4][0 0][0 1][0 2][0 3]]
# values=[1 3 4 9 2 8 5 7 2]
# shape=[2 6]


def get_train_model():
	x_inputs = tf.placeholder(tf.float32, [None, None, char_list_image_shape[0]])
	# inputs的维度是[batch_size,num_steps,input_dim]
	# 定义ctc_loss需要的标签向量
	targets = tf.sparse_placeholder(tf.int32)
	# 每个样本中有多少个时间序列
	seq_length = tf.placeholder(tf.int32, [None])
	# 定义LSTM网络的cell层，这里定义有num_hidden个单元
	cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
	# tf.contrib.rnn.MultiRNNCell是由多个简单的cells组成的RNN cell。用于构建多层循环神经网络。
	# state_is_tuple:如果为True，接受和返回的states是n-tuples，其中n=len(cells)。
	stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	# state_is_tuple=True时，我们是以LSTM为tf.nn.dynamic_rnn的输入cell类型；
	# state_is_tuple=False时，以GRU为tf.nn.dynamic_rnn的输入cell类型。
	# 如果cell选择了LSTM，那final_state是个tuple，分别代表Ct和ht，其中ht与outputs中的对应的最后一个时刻的输出ht相等；
	# 当cell为GRU时，state就只有一个了，原因是GRU将Ct和ht进行了简化，将其合并成了ht
	# 如果time_major == False(default)，输出张量形如[batch_size, max_time, cell.output_size]。
	# 如果time_major == True, 输出张量形如：[max_time, batch_size, cell.output_size]。
	# cell.output_size其实就是rnn cell中神经元的个数。
	outputs, _ = tf.nn.dynamic_rnn(cell, x_inputs, seq_length, dtype=tf.float32)
	# ->[batch_size,max_time_step,num_features]->lstm
	# ->[batch_size,max_time_step,cell.output_size]->reshape
	# ->[batch_size*max_time_step,num_hidden]->affine projection AW+b
	# ->[batch_size*max_time_step,num_classes]->reshape
	# ->[batch_size,max_time_step,num_classes]->transpose
	# ->[max_time_step,batch_size,num_classes]
	# 上面最后的shape就是标签向量的shape
	shape = tf.shape(x_inputs)
	# x_inputs的shape=[batch_size,image_shape[1],image_shape[0]]
	# 所以输入的数据是按列来排的，一列的像素为一个时间序列里输入的数据，一共120个时间序列
	batch_s, max_time_steps = shape[0], shape[1]
	# 输出的outputs为num_hidden个隐藏层单元的所有时刻的输出
	# reshape后的shape=[batch_size，num_hidden]
	outputs = tf.reshape(outputs, [-1, num_hidden])
	# 相当于一个全连接层，做一次线性变化
	w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="w")
	b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

	logits = tf.matmul(outputs, w) + b
	# 变换成和标签向量一致的shape
	logits = tf.reshape(logits, [batch_s, -1, num_classes])
	# logits的维度蒋欢，第1个维度和第0个维度互相交换
	logits = tf.transpose(logits, (1, 0, 2))

	return logits, x_inputs, targets, seq_length, w, b


def train():
	global_step = tf.Variable(0, trainable=False)
	# tf.train.exponential_decay函数实现指数衰减学习率
	learning_rate = tf.train.exponential_decay(lr_start, global_step, iteration, lr_decay_factor, staircase=True)
	logits, inputs, targets, seq_len, w, b = get_train_model()
	# 设置loss函数是ctc_loss函数
	# tf.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
	# labels: 是一个int32的SparseTensor, labels.indices[i, :] == [b, t]表示labels.values保存着(batch b, time t)的id。
	# inputs:一个3D Tensor(max_time * batch_size * num_classes).保存着logits.(通常是RNN接上一个线性神经元的输出)
	# sequence_length: 1D的int32向量, size为[batch_size]，是每个timestep中序列的长度。
	# 此sequence_length和用在dynamic_rnn中的sequence_length是一致的, 使用来表示rnn的哪些输出不是pad的.
	# preprocess_collapse_repeated: 设置为True的话, tensorflow会对输入的labels进行预处理, 连续重复的会被合成一个。
	# ctc_merge_repeated: 连续重复的是否被合成一个
	loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
	cost = tf.reduce_mean(loss)
	# CTC ：Connectionist Temporal Classifier 一般译为联结主义时间分类器 ，适合于输入特征和输出标签之间对齐关系不确定的时间序列问题
	# TC可以自动端到端地同时优化模型参数和对齐切分的边界。
	# 本例40X120大小的图片，切片成120列，输出标签最大设定为4(即不定长验证码最大长度为4),这样就可以用CTC模型进行优化。
	# 假设40x120的图片，数字串标签是"123"，把图片按列切分（CTC会优化切分模型），然后分出来的每块再去识别数字
	# 找出这块是每个数字或者特殊字符的概率（无法识别的则标记为特殊字符"-"）
	# 这样就得到了基于输入特征序列（图片）的每一个相互独立建模单元个体（划分出来的块）（包括“-”节点在内）的类属概率分布。
	# 基于概率分布，算出标签序列是"123"的概率P（123），当然这里设定"123"的概率为所有子序列之和，这里子序列包括'-'和'1'、'2'、'3'的连续重复

	# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
	# 这里用Adam算法来优化
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
	# 一种寻路策略，用于inference过程中,用于解码。
	decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

	acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

	def do_report():
		test_inputs, test_targets, test_seq_len = get_next_batch(test_batch_size, char_list_image_shape)
		test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
		dd, log_probs, accuracy = sess.run([decoded[0], log_prob, acc], test_feed)
		report_accuracy(dd, test_targets)

	def do_batch():
		train_inputs, train_targets, train_seq_len = get_next_batch(train_batch_size, char_list_image_shape)
		# 这里喂数据进行训练
		feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
		b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = sess.run(
			[loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

		print(b_cost, steps)
		if steps % report_step == 0:
			saver.save(sess, "./lstm_model/train_model", global_step=steps)
			do_report()
		return b_cost, steps

	if not os.path.exists("./lstm_model/"):
		os.mkdir("./lstm_model")
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# 训练num_epochs轮，每轮BATCHES个train_batch_size大小的样本
		if os.path.exists("./lstm_model/checkpoint"):
			# 判断模型是否存在，如果存在则从模型中恢复变量
			saver.restore(sess, tf.train.latest_checkpoint('./lstm_model/'))
		for epoch in range(num_epochs):
			train_cost = train_ler = 0
			for batch in range(BATCHES):
				start = time.time()
				# 每轮将一个batch的样本喂进去训练
				c, steps = do_batch()
				train_cost += c * train_batch_size
				seconds = time.time() - start
				print("Epoch:{:>05d},Step:{},batch seconds:{} ".format(epoch + 1, steps, seconds))

			train_cost /= TRAIN_SIZE

			train_inputs, train_targets, train_seq_len = get_next_batch(train_batch_size, char_list_image_shape)
			val_feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
			val_cost, val_ler, lr, steps = sess.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

			log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
			print(
				log.format(epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start,
						   lr))


if __name__ == '__main__':
	train()
