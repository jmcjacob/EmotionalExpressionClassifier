import cv2
import train
import argparse
from lbpmapping import LBP


def parse_args():
	desc = "A Real Time Emotional Classifier built using TensorFlow."
	parser = argparse.ArgumentParser(description=desc)

	# General
	parser.add_argument('--verbose', action='store_true', default=True,
						help='Boolean flag indicating if statements should be printed to console.')

	parser.add_argument('--log', type=str, default='',
						help='The file where files will be logged.')

	parser.add_argument('--resource_path', type=str, default='/Resources',
						help='Directory where models have/will be saved to.')

	parser.add_argument('--model_name', type=str, default='model',
						help='The name of the models will be loaded/saved in the resource directory.')

	# Training Parameters
	parser.add_argument('--train', action='store_true', default=False,
						help='Boolean flag indicating if the system will be training.')

	parser.add_argument('--training_dir', type=str, default='/Images',
						help='Directory of labeled images for training. Training only.')

	parser.add_argument('--epochs', type=int, default=1000,
						help='The number of epochs each model will be trained for. Training only.')

	parser.add_argument('--k_fold', type=int, default=5,
						help='The number of cross validation folds will be trained and testing. Training only.')

	# Classification
	parser.add_argument('--image', type=str,
						help='The path to an image that will be classified.')

	return parser.parse_args()


def log(args, input_str):
	if args.verbose:
		print input_str
	if args.log != '':
		file = open(args.log, 'a')
		file.write(input_str)
		file.close()


def main():
	global args
	args = parse_args()
	if args.train:
		train.train(args)
	else:
		pass # Classify

	lbp = LBP((256,256))
	cv2.imshow('Image', lbp.map_lbp('~/Downloads/LBP_mapping_Matlab/example_image.png', 1, 8))


if __name__ == '__main__':
	main()