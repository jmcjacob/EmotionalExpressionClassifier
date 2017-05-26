import os
import cv2
import main
import time
import argparse
import Classifier
import numpy as np


def train(args):
	start_time = time.clock()

	data = retrieve_data(args.training_dir)
	main.log(args, (time.clock() - start_time) + ' ' + str(len(data)) + ' images read across ' + str(len(data[0][1])) + ' classifications')

	



def retrieve_data(image_directory):
	data, label = [], 0
	folder_counter = sum([len(folder) for r, d, folder in os.walk(image_directory)])
	for folder in sorted(os.listdir(image_directory)):
		for image_file in os.listdir(image_directory + folder):
			labels = np.zeros(folder_counter)
			labels[label] = 1
			image = cv2.imread(image_directory + folder + '/' + image_file, cv2.IMREAD_COLOR)
			data.append((cv2.resize(image, (150, 150)), label))
		label += 1
	return data
