import os
import cv2
import main
import time
import numpy as np
from classifier import Classifier


def train(args):
	start_time = time.clock()

	main.log(args, '\n---------- RGB, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/rgb')
	test_data = retrieve_data(args.testing_dir + '/rgb')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + ' ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	rgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=False)
	accuracy = rgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'rgb'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- LBP, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/lbp')
	test_data = retrieve_data(args.testing_dir + '/lbp')
	main.log(args, str(time.clock() - start_time) + ' ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	lbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=False)
	accuracy = lbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'lbp'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- RGB, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/frgb')
	test_data = retrieve_data(args.testing_dir + '/frgb')
	main.log(args, str(time.clock() - start_time) + ' ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	frgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=False)
	accuracy = frgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'frgb'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- RGB, Frontalization, Locally Connected Layers ----------')
	lfrgb_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=True)
	accuracy = lfrgb_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'lfrgb'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')


	main.log(args, '\n---------- LBP, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/flbp')
	test_data = retrieve_data(args.testing_dir + '/flbp')
	main.log(args, str(time.clock() - start_time) + ' ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	flbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=False)
	accuracy = flbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'flbp'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')

	main.log(args, '\n---------- LBP, Frontalization, Locally Connected Layers ----------')
	lflbp_classifier = Classifier(args, start_time, len(train_data[0][1]), args.resource_path, train_data[0][0].shape, local=True)
	accuracy = lflbp_classifier.train(train_data, test_data, args.epochs, (args.model_name + 'lflbp'))
	main.log(args, str(time.clock() - start_time) + ' ' + str(accuracy) + ' accuracy')

	main.log(args, '\n Completed Training in ' + str(time.clock() - start_time) + 's')

	accuracy(args, [rgb_classifier, lbp_classifier, frgb_classifier, lfrgb_classifier, flbp_classifier, lflbp_classifier])

def retrieve_data(image_directory):
	data, label = [], 0
	folder_counter = sum([len(folder) for r, d, folder in os.walk(image_directory)])
	folder_counter = 8
	for folder in sorted(os.listdir(image_directory)):
		for image_file in sorted(os.listdir(image_directory + '/' +  folder)):
			labels = np.zeros(folder_counter)
			labels[label] = 1
			image_file = image_directory + '/' + folder + '/' + image_file
			image = cv2.imread(image_file, cv2.IMREAD_COLOR)
			data.append((cv2.resize(image, (150, 150)), labels))
		label += 1
	return data


def accuracy(args, classifiers):
	rgb, lbp = retrieve_data(args.testing_dir + '/rgb'), retrieve_data(args.testing_dir + '/lbp')
	frgb, flbp = retrieve_data(args.testing_dir + '/front_rgb'), retrieve_data(args.testing_dir + '/front_lbp')
	labels, _y = np.zeros(0), []

	_y += classifiers[0].classify_batch(rgb, (args.model_name + 'rgb'))[1]
	_y += classifiers[1].classify_batch(lbp, (args.model_name + 'lbp'))[1]
	_y += classifiers[2].classify_batch(frgb, (args.model_name + 'frgb'))[1]
	_y += classifiers[3].classify_batch(frgb, (args.model_name + 'lfrgb'))[1]
	_y += classifiers[4].classify_batch(flbp, (args.model_name + 'flbp'))[1]
	results = classifiers[5].classify_batch(flbp, (args.model_name + 'lflbp'))
	_y += results[1]
	labels = results[0]

	classifiers[0].confusion_matrix(labels, _y)
