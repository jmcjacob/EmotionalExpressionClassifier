import os
import cv2
import main
import time
import tflearn
import numpy as np
import tensorflow as tf
from classifier import Classifier


def train_2(args):

	# Load data
	data = retrieve_data(args.training_dir + '/rgb')
	data += retrieve_data(args.testing_dir + '/rgb')
	x, y = [m[0] for m in data], [n[1] for n in data]

	# Define number of output classes.
	num_classes = 7

	# Define padding scheme.
	padding = 'VALID'

	# Model Architecture
	network = tflearn.input_data(shape=[None, 224, 224, 1])
	conv_1 = tflearn.relu(tflearn.conv_2d(network, 64, 7, strides=2, bias=True, padding=padding, activation=None, name='Conv2d_1'))
	maxpool_1 = tflearn.batch_normalization(tflearn.max_pool_2d(conv_1, 3, strides=2, padding=padding, name='MaxPool_1'))

	# FeatEX-1
	conv_2a = tflearn.relu(tflearn.conv_2d(maxpool_1, 96, 1, strides=1, padding=padding, name='Conv_2a_FX1'))
	maxpool_2a = tflearn.max_pool_2d(maxpool_1, 3, strides=1, padding=padding, name='MaxPool_2a_FX1')
	conv_2b = tflearn.relu(tflearn.conv_2d(conv_2a, 208, 3, strides=1, padding=padding, name='Conv_2b_FX1'))
	conv_2c = tflearn.relu(tflearn.conv_2d(maxpool_2a, 64, 1, strides=1, padding=padding, name='Conv_2c_FX1'))
	FX1_out = tflearn.merge([conv_2b, conv_2c], mode='concat', axis=3, name='FX1_out')
	# FeatEX-2
	conv_3a = tflearn.relu(tflearn.conv_2d(FX1_out, 96, 1, strides=1, padding=padding, name='Conv_3a_FX2'))
	maxpool_3a = tflearn.max_pool_2d(FX1_out, 3, strides=1, padding=padding, name='MaxPool_3a_FX2')
	conv_3b = tflearn.relu(tflearn.conv_2d(conv_3a, 208, 3, strides=1, padding=padding, name='Conv_3b_FX2'))
	conv_3c = tflearn.relu(tflearn.conv_2d(maxpool_3a, 64, 1, strides=1, padding=padding, name='Conv_3c_FX2'))
	FX2_out = tflearn.merge([conv_3b, conv_3c], mode='concat', axis=3, name='FX2_out')
	net = tflearn.flatten(FX2_out)
	loss = tflearn.fully_connected(net, num_classes, activation='softmax')

	# Compile the model and define the hyperparameters
	network = tflearn.regression(loss, optimizer='Adam',loss='categorical_crossentropy',learning_rate=0.0001)

	# Final definition of model checkpoints and other configurations
	model = tflearn.DNN(network, checkpoint_path=args.resource_dir + args.model_name,max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir=args.log)

	# Fit the model, train for 20 epochs. (Change all parameters to flags (arguments) on version 2.)
	model.fit(x, y, n_epoch=20, validation_set=0.1, shuffle=True, show_metric=True, batch_size=350, snapshot_step=2000, snapshot_epoch=True, run_id=args.model_name)

	# Save the model
	model.save(args.resource_dir + args.model_name + 'thing.model')


def train(args):
	start_time = time.clock()
	main.log(args, '\n---------- RGB, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/rgb')
	test_data = retrieve_data(args.testing_dir + '/rgb')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	rgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'rgb', local=False)
	accuracy = rgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/rgb/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- LBP, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/lbp')
	test_data = retrieve_data(args.testing_dir + '/lbp')
	main.log(args, str(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	lbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'lbp', local=False)
	accuracy = lbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/lbp/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- RGB, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/frgb')
	test_data = retrieve_data(args.testing_dir + '/frgb')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	frgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'frgb', local=False)
	accuracy = frgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/frgb/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- RGB, Frontalization, Locally Connected Layers ----------')
	lfrgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'lfrgb', local=True)
	accuracy = lfrgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/lfrgb/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- LBP, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/flbp')
	test_data = retrieve_data(args.testing_dir + '/flbp')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	flbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'flbp', local=False)
	accuracy = flbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/flbp/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')

	main.log(args, '\n---------- LBP, Frontalization, Locally Connected Layers ----------')
	lflbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, 'lflbp', local=True)
	accuracy = lflbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + '/lflbp/'))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(accuracy) + ' accuracy')

	main.log(args, '\n Completed Training in ' + str(time.clock() - start_time) + 's\n')

	avg_accuracy(args, start_time)

def retrieve_data(image_directory):
	data, label = [], 0
	for _, dirs, _ in os.walk(image_directory):
		folder_counter = len(dirs)
		break
	for folder in sorted(os.listdir(image_directory)):
		for image_file in sorted(os.listdir(image_directory + '/' +  folder)):
			try:
				labels = np.zeros(folder_counter)
				labels[label] = 1
				image_file = image_directory + '/' + folder + '/' + image_file
				image = cv2.imread(image_file, cv2.IMREAD_COLOR)
				data.append((cv2.resize(image, (150, 150)), labels))
			except:
				pass
		label += 1
	return data


def avg_accuracy(args, start_time):
	rgb, lbp = retrieve_data(args.testing_dir + '/rgb'), retrieve_data(args.testing_dir + '/lbp')
	frgb, flbp = retrieve_data(args.testing_dir + '/frgb'), retrieve_data(args.testing_dir + '/flbp')
	labels, predictions = np.zeros(0), []

	results = Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'rgb', local=False).classify_batch(rgb, (args.model_name + '/rgb/'))
	predictions.append(results[1])
	labels = results[0]
	predictions.append(Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'lbp', local=False).classify_batch(lbp, (args.model_name + '/lbp/'))[1])
	predictions.append(Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'frgb', local=False).classify_batch(frgb, (args.model_name + '/frgb/'))[1])
	predictions.append(Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'lfrgb', local=True).classify_batch(frgb, (args.model_name + '/lfrgb/'))[1])
	predictions.append(Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'flbp', local=False).classify_batch(flbp, (args.model_name + '/flbp/'))[1])
	predictions.append(Classifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, 'lflbp', local=True).classify_batch(flbp, (args.model_name + '/lflbp/'))[1])

	votes = np.zeros((len(predictions[0]), len(labels[0])))
	for i in range(len(predictions)):
		for j in range(len(predictions[0])):
			vote = predictions[i][j]
			votes[j][int(vote)] += 1.0

	prediction = []
	for i in range(len(votes)):
		prediction.append(np.argmax(votes[i]))

	Classifier.confusion_matrix(args, prediction, labels)
