import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


class Classifier:
	def __init__(self, num_classes, save_path, input_shape, colour=False, local=False):
		if colour:
			self.x = tf.placeholder('float', [None, input_shape[0], input_shape[1], input_shape[2]])
		else:
			self.x = tf.placeholder('float', [None, input_shape[0], input_shape[1]])
		self.y = tf.placeholder('float', [None, num_classes])
		self.keep_prob = tf.placeholder(tf.float32)
		self.model = self.build_model(num_classes, input_shape, colour, local)
		self.save_path = save_path

	def build_model(self, num_classes, input_shape, colour, local):
		if colour:
			model = tf.reshape(self.x, shape=[-1, input_shape[0], input_shape[1], input_shape[2]])
		else:
			model = tf.reshape(self.x, shape=[-1, input_shape[0], input_shape[1], 1])
		model = tf.pad(model, [1, 3, 3, 1], 'CONSTANT')
		if local:
			model = tf.nn.relu(local(model, 7, 64, [1, 3, 3, 1], 'VALID', 'Local_w', 'Local_b'))
		else:
			model = tf.nn.conv2d(model, tf.Variable(tf.random_normal([7, 7, 1, 64])), [1, 3, 3, 1], 'VALID')
			model = tf.nn.relu(tf.nn.bias_add(model, tf.Variable(tf.random_normal([64]))))
		model = tf.nn.max_pool(model, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
		model = tf.nn.lrn(model)
		model = self.featex(self.featex(model))
		shape = len(model[1] * model[2] * model[3])
		return tf.matmul(tf.reshape(model, (-1, shape)), tf.Variable(tf.random_normal([shape, num_classes])))

	@staticmethod
	def local(previous_layer, kernel_size, channels, strides, padding, weight_name, bias_name):
		shape = previous_layer.get_shape()
		height = shape[1].value
		width = shape[2].value
		patch_size = (kernel_size ** 2) * shape[3].value
		patches = tf.extract_image_patches(previous_layer, [1, kernel_size, kernel_size, 1],
										strides, [1, 1, 1, 1], padding)
		weights = tf.get_variable(weight_name, [1, height, width, patch_size, channels],
								initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
		biases = tf.get_variable(bias_name, [height, width, channels], initializer=tf.constant_initializer(0.1))
		mul = tf.multiply(tf.expand_dims(patches, axis=-1), weights)
		ssum = tf.reduce_sum(mul, axis=3)
		return tf.add(ssum, biases)

	@staticmethod
	def featex(model):
		weights = {
			'wca': tf.Variable(tf.random_normal([1, 1, 64, 96])),
			'wcb': tf.Variable(tf.random_normal([3, 3, 96, 208])),
			'wcc': tf.Variable(tf.random_normal([1, 1, 64, 64])),
		}
		biases = {
			'bca': tf.Variable(tf.random_normal([96])),
			'bcb': tf.Variable(tf.random_normal([208])),
			'bcc': tf.Variable(tf.random_normal([64]))
		}

		path1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model, weights['wca'], [1, 1, 1, 1], 'VALID'), biases['bca']))
		path1 = tf.pad(path1, [1, 1, 1, 1], 'CONSTANT')
		path1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(path1, weights['wcb'], [1, 1, 1, 1], 'VALID'), biases['bcb']))

		path2 = tf.pad(model, [1, 1, 1, 1], 'CONSTANT')
		path2 = tf.nn.max_pool(path2, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
		path2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(path2, weights['wcc'], [1, 1, 1, 1], 'VALID'), biases['bcc']))

		output = tf.concat([path1, path2], 0)
		return tf.nn.max_pool(output, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

	def train(self, training_data, testing_data, epochs, log, batch_size=32, intervals=1):
		batches = self.split_data(training_data, batch_size)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)
		correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		init = tf.global_variables_initializer()
		count = self.count_trainable_vars()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			summary_writer = tf.summary.FileWriter(self.save_path + log, graph=tf.get_default_graph())
			for epoch in range(epochs):
				avg_loss, avg_acc = 0, 0
				summary = tf.Summary()
				for i in range(len(batches)):
					x, y = [m[0] for m in batches[i]], [n[1] for n in batches[i]]
					_, loss, acc = sess.run([optimizer, cost, accuracy], feed_dict={self.x: x, self.y: y, self.keep_prob: .5})
					avg_loss += loss
					avg_acc += acc
					summary.value.add(tag='Accuracy', simple_value=(avg_acc / len(batches)))
					summary.value.add(tag='Loss', simple_value=(avg_loss / len(batches)))
				summary_writer.add_summary(summary, epoch)
				if epoch % intervals == 0 and intervals != 0:
					print log, 'Epoch', '%03d' % (epoch + 1), ' Loss = {:.5f}'.format(avg_loss / len(batches))
			saver.save(sess, self.save_path + log + 'model') if self.save_path != '' else ''
			batches = self.split_data(testing_data, batch_size)
			avg_acc, labels, _y = 0, np.zeros(0), []
			for batch in batches:
				x, y = [m[0] for m in batch], [n[1] for n in batch]
				_y += y
				acc = accuracy.eval({self.x: x, self.y: y, self.keep_prob: 1.})
				avg_acc += acc / len(batches)
				prediction = tf.argmax(self.model, 1)
				label = prediction.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.}, session=sess)
				labels = np.append(labels, label)
			return avg_acc, _y, labels, count

	def cross_validation(self, data, k, epochs, log):
		start_time = time.clock()
		k_fold = KFold(n_splits=k)
		i, acc, count = 0, 0, 0
		_y, labels = [], []
		for train_indices, test_indices in k_fold.split(data):
			a, b, c, count = self.train(train_indices, test_indices, epochs, log + str(i))
			acc += (a / k)
			_y += b
			labels += c
			i += 1
		print 'Total Time: ' + str(time.clock() - start_time)
		print 'Average Accuracy: ' + str(acc)
		self.confusion_matrix(labels, _y)

	@staticmethod
	def split_data(seq, num):
		count, out = -1, []
		while count < len(seq):
			temp = []
			for i in range(num):
				count += 1
				if count >= len(seq):
					break
				temp.append(seq[count])
			if len(temp) != 0:
				out.append(temp)
		return out

	@staticmethod
	def count_trainable_vars():
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parametes = 1
			for dim in shape:
				variable_parametes *= dim.value
			total_parameters += variable_parametes
		return total_parameters

	@staticmethod
	def confusion_matrix(labels, y_pred):
		y_actu = np.zeros(len(labels))
		for i in range(len(labels)):
			for j in range(len(labels[i])):
				if labels[i][j] == 1.0:
					labels[i] = int(j)

		p_labels = pd.Series(y_pred)
		t_labels = pd.Series(y_actu)
		df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

		print '\n', df_confusion
		print '\n', classification_report(y_actu, y_pred)

	def classify(self, data, model):
		init, saver = tf.global_variables_initializer(), tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, self.save_path + model)
			return np.asarray(sess.run(self.model, feed_dict={self.x: data, self.keep_prob: 1.}))
