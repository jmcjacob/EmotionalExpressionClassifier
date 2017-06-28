import main
import time
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from tflearn.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.layers.local import LocallyConnected2D


class MonitorCallback(Callback):
	def __init__(self, args, start_time):
		super(self.__class__, self).__init__()
		self.args = args
		self.start = start_time
		self.log = args.model_name

	def on_epoch_end(self, state):
		main.log(self.args, '{:.5f}s Epoch '.format(time.clock() - self.start) + str(state.epoch).zfill(4) + ' Loss = {:.5f}'.format(state.global_loss) + ' Accuracy = {:.5}'.format(state.acc_value))

	def on_train_end(self, state):
		main.log(self.args, '\n{:.5f}s'.format(time.clock() - self.start) + ' Validation Loss = {:.5f}'.format(state.val_loss) + ' Validation Accuracy = ' + str(state.val_acc))


class Classifier:
	def __init__(self, args, start_time, num_classes, save_path, input_shape, scope_name, model):
		tf.reset_default_graph()
		self.args = args
		self.scope_name = scope_name
		self.start_time = start_time
		self.save_path = self.args.resource_dir + self.args.model_name + self.scope_name
		network = self.build_network(num_classes, input_shape, model)
		self.model = tflearn.DNN(network, checkpoint_path=self.save_path, max_checkpoints=1, tensorboard_verbose=3, tensorboard_dir=self.save_path + 'tensorboard')

	def load(self):
		self.model.load(self.save_path + 'model.model')

	@staticmethod
	def local(input, nb_filters, filter_size, stride, name):
		local = LocallyConnected2D(nb_filters, filter_size, strides=(stride, stride), use_bias=True, kernel_initializer='random_normal', bias_initializer='random_normal', name=name)
		local.build(input.get_shape().as_list())
		return local.call(input)

	def build_network(self, num_classes, input_shape, model):
		network = tflearn.input_data(shape=[None, input_shape[0], input_shape[1], input_shape[2]])
		if model == 'DeepFace':
			conv_1 = tflearn.relu(tflearn.conv_2d(network, 32, 11, strides=1, padding='VALID', name='Conv2d_1'))
			maxpool_1 = tflearn.max_pool_2d(conv_1, 3, strides=2, padding='VALID', name='MaxPool_1')
			conv_2 = tflearn.relu(tflearn.conv_2d(maxpool_1, 32, 9, strides=1, padding='VALID', name='Conv2d_2'))

			local_1 = tflearn.relu(self.local(conv_2, 16, 9, 1, 'Local_1'))
			local_2 = tflearn.relu(self.local(local_1, 16, 7, 1, 'Local_2'))
			local_3 = tflearn.relu(self.local(local_2, 16, 5, 1, 'Local_3'))

			flatterned = tflearn.flatten(local_3)
			full_1 = tflearn.dropout(tflearn.relu(tflearn.fully_connected(flatterned, 4096, name='Fully_Connected_1')), 0.5)
			output = tflearn.fully_connected(full_1, num_classes, activation='softmax', name='Output')

		elif model == 'Song':
			conv_1 = tflearn.conv_2d(network, 64, 5, strides=1, padding='VALID', name='Conv_1')
			maxpool_1 = tflearn.max_pool_2d(conv_1, 3, strides=2, padding='VALID', name='MaxPool_1')
			conv_2 = tflearn.conv_2d(maxpool_1, 64 , 5, strides=1, padding='VALID', name='Conv_2')
			maxpool_2 = tflearn.max_pool_2d(conv_2, 3, strides=2, padding='VALID', name='MaxPool_2')
			local_1 = tflearn.dropout(self.local(maxpool_2, 32, 3, 1, 'Local_1'), 0.5)
			local_2 = tflearn.dropout(self.local(local_1, 32, 3, 1, 'Local_2'), 0.5)
			output = tflearn.fully_connected(tflearn.flatten(local_2), num_classes, activation='softmax', name='Output')

		else:
			conv_1 = tflearn.relu(tflearn.conv_2d(network, 64, 7, strides=2, bias=True, padding='VALID', name='Conv2d_1'))
			maxpool_1 = tflearn.batch_normalization(tflearn.max_pool_2d(conv_1, 3, strides=2, padding='VALID', name='MaxPool_1'))

			conv_2a = tflearn.relu(tflearn.conv_2d(maxpool_1, 96, 1, strides=1, padding='VALID', name='Conv_2a_FX1'))
			maxpool_2a = tflearn.max_pool_2d(maxpool_1, 3, strides=1, padding='VALID', name='MaxPool_2a_FX1')
			conv_2b = tflearn.relu(tflearn.conv_2d(conv_2a, 208, 3, strides=1, padding='VALID', name='Conv_2b_FX1'))
			conv_2c = tflearn.relu(tflearn.conv_2d(maxpool_2a, 64, 1, strides=1, padding='VALID', name='Conv_2c_FX1'))
			FX1_out = tflearn.merge([conv_2b, conv_2c], mode='concat', axis=3, name='FX1_out')

			conv_3a = tflearn.relu(tflearn.conv_2d(FX1_out, 96, 1, strides=1, padding='VALID', name='Conv_3a_FX2'))
			maxpool_3a = tflearn.max_pool_2d(FX1_out, 3, strides=1, padding='VALID', name='MaxPool_3a_FX2')
			conv_3b = tflearn.relu(tflearn.conv_2d(conv_3a, 208, 3, strides=1, padding='VALID', name='Conv_3b_FX2'))
			conv_3c = tflearn.relu(tflearn.conv_2d(maxpool_3a, 64, 1, strides=1, padding='VALID', name='Conv_3c_FX2'))
			FX2_out = tflearn.merge([conv_3b, conv_3c], mode='concat', axis=3, name='FX2_out')
			net = tflearn.flatten(FX2_out)
			output = tflearn.fully_connected(net, num_classes, activation='softmax', name='Output')

		return tflearn.regression(output, optimizer='Adam', loss='categorical_crossentropy', learning_rate=0.0001)

	def train(self, training_data, testing_data):
		x, y = [m[0] for m in training_data], [n[1] for n in training_data]
		monitor = MonitorCallback(self.args, self.start_time)
		self.model.fit(x, y, n_epoch=self.args.epochs, validation_set=0.1, shuffle=True, show_metric=True, batch_size=self.args.batch_size, snapshot_step=2000, snapshot_epoch=True,
					   run_id=self.args.model_name, callbacks=monitor)
		main.log(self.args, '{:.5f}s '.format(time.clock() - self.start_time) + str(self.count_trainable_vars()) + ' trainable parameters')
		self.model.save(self.save_path + 'model.model')
		predictions, labels = self.evaluate(testing_data)
		self.confusion_matrix(self.args, predictions, labels)

	def classify(self, data):
		return self.model.predict(data)

	def evaluate(self, testing_data):
		predictions = []
		for data in testing_data:
			predictions += (self.model.predict(data[0]), data[1])
		return [m[0] for m in predictions], [n[1] for n in predictions]

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
	def confusion_matrix(args, predictions, labels):
		y_actu = np.zeros(len(labels))
		for i in range(len(labels)):
			for j in range(len(labels[i])):
				if labels[i][j] == 1.00:
					y_actu[i] = j
		y_pred = np.zeros(len(predictions))
		for i in range(len(predictions)):
			y_pred[i] = np.argmax(predictions[i])

		p_labels = pd.Series(y_pred)
		t_labels = pd.Series(y_actu)
		df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
		main.log(args, '\nAccuracy = ' + str(accuracy_score(y_true=y_actu, y_pred=y_pred, normalize=True)) + '\n')
		main.log(args, df_confusion)
		main.log(args, ' ')
		main.log(args, classification_report(y_actu, y_pred))

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
