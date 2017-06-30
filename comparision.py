import os
import cv2
import main
import time
import numpy as np
from classifier import Classifier


def train(args):
	start_time = time.clock()
	rgb_data = [retrieve_data(args.training_dir + '/rgb'), retrieve_data(args.testing_dir + '/rgb')]
	front_data = [retrieve_data(args.training_dir + '/frgb'), retrieve_data(args.testing_dir + '/frgb')]

	main.log(args, '\n---------- DeXpression, No Frontalization ----------')
	Dexrgb = Classifier(args, start_time, len(rgb_data[0][0][1]), args.resource_dir, rgb_data[0][0][0].shape, '/Dexrgb/', 'DeXpression')
	Dexrgb.train(rgb_data[0], rgb_data[1])

	main.log(args, '\n---------- DeXpression, Frontalization ----------')
	Dexfront = Classifier(args, start_time, len(front_data[0][0][1]), args.resource_dir, front_data[0][0][0].shape, '/Dexfront/', 'DeXpression')
	Dexfront.train(front_data[0], front_data[1])

	main.log(args, '\n---------- DeepFace, No Frontalization ----------')
	Deeprgb = Classifier(args, start_time, len(rgb_data[0][0][1]), args.resource_dir, rgb_data[0][0][0].shape, '/Deeprgb/', 'DeepFace')
	Deeprgb.train(rgb_data[0], rgb_data[1])

	main.log(args, '\n---------- DeepFace, Frontalization ----------')
	Deepfront = Classifier(args, start_time, len(front_data[0][0][1]), args.resource_dir, front_data[0][0][0].shape, '/Deepfront/', 'DeepFace')
	Deepfront.train(front_data[0], front_data[1])

	main.log(args, '\n---------- Song, No Frontalization ----------')
	Songrgb = Classifier(args, start_time, len(rgb_data[0][0][1]), args.resource_dir, rgb_data[0][0][0].shape, '/Songrgb/', 'Song')
	Songrgb.train(rgb_data[0], rgb_data[1])

	main.log(args, '\n---------- Song, Frontalization ----------')
	Songfront = Classifier(args, start_time, len(front_data[0][0][1]), args.resource_dir, front_data[0][0][0].shape, '/Songfront/', 'Song')
	Songfront.train(front_data[0], front_data[1])

	main.log(args, '\n Completed Comparison in ' + str(time.clock() - start_time) + 's\n')


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
				data.append((cv2.resize(image, (88, 88)), labels))
			except:
				pass
		label += 1
	return data
