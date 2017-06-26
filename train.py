import os
import cv2
import main
import time
import numpy as np
from classifier import Classifier
from newClassifier import Classifier as newClassifier


def train_2(args):
	start_time = time.clock()
	main.log(args, '\n---------- RGB, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/rgb')
	test_data = retrieve_data(args.testing_dir + '/rgb')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	rgb_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/rgb/', local=False)
	rgb_classifier.train(train_data, test_data)

	main.log(args, '\n---------- LBP, No Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/lbp')
	test_data = retrieve_data(args.testing_dir + '/lbp')
	main.log(args, str(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	lbp_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/lbp/', local=False)
	lbp_classifier.train(train_data, test_data)

	main.log(args, '\n---------- RGB, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/frgb')
	test_data = retrieve_data(args.testing_dir + '/frgb')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	frgb_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/frgb/', local=False)
	frgb_classifier.train(train_data, test_data)

	main.log(args, '\n---------- RGB, Frontalization, Locally Connected Layers ----------')
	lfrgb_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/lfrgb/', local=True)
	lfrgb_classifier.train(train_data, test_data)

	main.log(args, '\n---------- LBP, Frontalization, Convolutional Layers ----------')
	train_data = retrieve_data(args.training_dir + '/flbp')
	test_data = retrieve_data(args.testing_dir + '/flbp')
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(train_data)) + ' images read across ' + str(len(train_data[0][1])) + ' classes')

	flbp_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/flbp/', local=False)
	flbp_classifier.train(train_data, test_data)

	main.log(args, '\n---------- LBP, Frontalization, Locally Connected Layers ----------')
	lflbp_classifier = newClassifier(args, start_time, len(train_data[0][1]), args.resource_dir, train_data[0][0].shape, '/lflbp/', local=True)
	lflbp_classifier.train(train_data, test_data)

	main.log(args, '\n Completed Training in ' + str(time.clock() - start_time) + 's\n')

	avg_accuracy_2(args, start_time)


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


def avg_accuracy_2(args, start_time):
	rgb, lbp = retrieve_data(args.testing_dir + '/rgb'), retrieve_data(args.testing_dir + '/lbp')
	frgb, flbp = retrieve_data(args.testing_dir + '/frgb'), retrieve_data(args.testing_dir + '/flbp')
	labels, predictions = np.zeros(0), []

	results = newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/rgb/', local=False).evaluate(rgb)
	predictions.append(results[0])
	labels = results[1]
	predictions.append(newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/lbp/', local=False).evaluate(lbp)[0])
	predictions.append(newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/frgb/', local=False).evaluate(frgb)[0])
	predictions.append(newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/lfrgb/', local=True).evaluate(frgb)[0])
	predictions.append(newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/flbp/', local=False).evaluate(flbp)[0])
	predictions.append(newClassifier(args, start_time, len(rgb[0][1]), args.resource_dir, rgb[0][0].shape, '/lflbp/', local=True).evaluate(flbp)[0])

	votes = np.zeros((len(predictions[0]), len(labels[0])))
	for i in range(len(predictions)): # For each Classifier
		for j in range(len(predictions[0])): # For each image
			vote = np.argmax(predictions[i][j])
			votes[j][int(vote)] += 1.0

	prediction = []
	for i in range(len(votes)):
		prediction.append(np.argmax(votes[i]))

	newClassifier.confusion_matrix(args, votes, labels)
