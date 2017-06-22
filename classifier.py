import main
import time
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers.local import LocallyConnected2D
from sklearn.metrics import classification_report


class Classifier:
	def __init__(self, args, start_time, num_classes, save_path, input_shape, scope_name, local=False):
		tf.reset_default_graph()
		self.args = args
		self.scope_name = scope_name
		self.start_time = start_time
		self.x = tf.placeholder('float', [None, input_shape[0], input_shape[1], input_shape[2]])
		self.y = tf.placeholder('float', [None, num_classes])
		self.keep_prob = tf.placeholder(tf.float32)
		self.model = self.build_model(num_classes, input_shape, local)
		self.save_path = save_path

	def build_model(self, num_classes, input_shape, local):
		with tf.variable_scope(self.scope_name):
			model = tf.reshape(self.x, shape=[-1, input_shape[0], input_shape[1], input_shape[2]])
			model = tf.pad(model, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
			if local:
				#model = tf.nn.relu(self.local_layer(model, 7, 64, [1, 1, 1, 1], 'SAME', 'Local_w', 'Local_b'))
				local = LocallyConnected2D(64, 7, (3, 3), 'valid', activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal')
				local.build(model.get_shape())
				model = local.call(model)
			else:
				model = tf.nn.conv2d(model, tf.Variable(tf.random_normal([7, 7, input_shape[2], 64])), [1, 3, 3, 1], 'VALID')
				model = tf.nn.relu(tf.nn.bias_add(model, tf.Variable(tf.random_normal([64]))))
			model = tf.nn.max_pool(model, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
			model = tflearn.batch_normalization(model)
			model = self.featex(self.featex(model))
			model = tflearn.flatten(model)
			return tf.matmul(model, tf.Variable(tf.random_normal([model.get_shape().as_list[1], num_classes])))

	@staticmethod
	def local_layer(previous_layer, kernel_size, channels, strides, padding, weight_name, bias_name):
		shape = previous_layer.get_shape()
		height = shape[1].value
		width = shape[2].value
		patch_size = (kernel_size ** 2) * shape[3].value
		patches = tf.extract_image_patches(previous_layer, [1, kernel_size, kernel_size, 1], strides, [1, 1, 1, 1], padding)
		with tf.device('/cpu:0'):
			weights = tf.get_variable(weight_name, [1, height, width, patch_size, channels], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16))
			biases = tf.get_variable(bias_name, [height, width, channels], initializer=tf.constant_initializer(0.1))
		mul = tf.multiply(tf.expand_dims(patches, axis=-1), weights)
		ssum = tf.reduce_sum(mul, axis=3)
		return tf.add(ssum, biases)

	@staticmethod
	def featex(model):
		weights = {
			'wca': tf.Variable(tf.random_normal([1, 1, model.get_shape().as_list()[3], 96])),
			'wcb': tf.Variable(tf.random_normal([3, 3, 96, 64])),
			'wcc': tf.Variable(tf.random_normal([1, 1, model.get_shape().as_list()[3], 64])),
		}
		biases = {
			'bca': tf.Variable(tf.random_normal([96])),
			'bcb': tf.Variable(tf.random_normal([64])),
			'bcc': tf.Variable(tf.random_normal([64]))
		}

		path1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(model, weights['wca'], [1, 1, 1, 1], 'VALID'), biases['bca']))
		path1 = tf.pad(path1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		path1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(path1, weights['wcb'], [1, 1, 1, 1], 'VALID'), biases['bcb']))

		path2 = tf.pad(model, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		path2 = tf.nn.max_pool(path2, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
		path2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(path2, weights['wcc'], [1, 1, 1, 1], 'VALID'), biases['bcc']))

		output = tflearn.merge([path1, path2], mode='concat', axis=3)
		return tf.nn.max_pool(output, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

	def train(self, training_data, testing_data, epochs, log, intervals=1):
		with tf.variable_scope(self.scope_name):
			batch_size = self.args.batch_size
			batches = self.split_data(training_data, batch_size)
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
			optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
			correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

			init = self.init_scope()
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
					if epoch % intervals == 0 and intervals != 0 and self.args.verbose:
						main.log(self.args, '{:.5f}'.format(time.clock() - self.start_time) + 's ' + str(log) + ' Epoch ' + str(epoch + 1) + ' Loss = {:.5f}'.format(avg_loss / len(batches)))
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
				main.log(self.args, '{:.5f}'.format(time.clock() - self.start_time) + 's ' + str(count) + ' trainable parameters')
				if self.args.verbose:
					self.confusion_matrix(self.args, labels, _y)
				return avg_acc

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
	def confusion_matrix(args, y_pred, labels):
		y_actu = np.zeros(len(y_pred))
		for i in range(len(labels)):
			for j in range(len(labels[i])):
				if labels[i][j] == 1.00:
					y_actu[i] = j

		p_labels = pd.Series(y_pred)
		t_labels = pd.Series(y_actu)
		df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

		main.log(args, df_confusion)
		main.log(args, classification_report(y_actu, y_pred))

	def classify(self, data, log):
		with tf.variable_scope(self.scope_name):
			init, saver = self.init_scope(), tf.train.Saver()
			with tf.Session() as sess:
				sess.run(init)
				saver.restore(sess, self.save_path + log + 'model')
				return np.asarray(sess.run(self.model, feed_dict={self.x: data, self.keep_prob: 1.}))

	def classify_batch(self, data, log):
		with tf.variable_scope(self.scope_name):
			batch_size = self.args.batch_size
			init, saver = self.init_scope(), tf.train.Saver()
			batches = self.split_data(data, batch_size)
			labels, _y = np.zeros(0), []
			with tf.Session() as sess:
				sess.run(init)
				saver.restore(sess, self.save_path + log + 'model')
				for batch in batches:
					x, y = [m[0] for m in batch], [n[1] for n in batch]
					_y += y
					prediction = tf.argmax(self.model, 1)
					label = prediction.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.}, session=sess)
					labels = np.append(labels, label)
				return _y, labels

	def init_scope(self):
		varibles = []
		for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name):
			varibles.append(i)
		return tf.variables_initializer(varibles)
