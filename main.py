import train
import argparse
import run
from classify import classify
from dataset import build_dataset


def parse_args():
	desc = "A Real Time Emotional Classifier built using TensorFlow."
	parser = argparse.ArgumentParser(description=desc)

	# General
	parser.add_argument('--mode', type=str,
						help='Model for the system, \'train\', \'classify\', \'dataset\', \'run\'')

	parser.add_argument('--verbose', action='store_true', default=True,
						help='Boolean flag indicating if statements should be printed to console.')

	parser.add_argument('--log', type=str, default='',
						help='The file where files will be logged.')

	parser.add_argument('--resource_dir', type=str, default='/Resources',
						help='Directory where models will be saved to and necessary data will be stored.')

	parser.add_argument('--model_name', type=str, default='model',
						help='The name of the models will be loaded/saved in the resource directory.')

	# Training Parameters
	parser.add_argument('--training_dir', type=str, default='/Training_data',
						help='Directory of labeled images for training. Training and Dataset Building only.')

	parser.add_argument('--testing_dir', type=str, default='/Testing_data',
						help='Directory of labeled images for testing. Training and Dataset Building only.')

	parser.add_argument('--epochs', type=int, default=1000,
						help='The number of epochs each model will be trained for. Training only.')

	parser.add_argument('--batch_size', type=int, default=32,
						help='The sizes of the batches used within training. Training only.')

	# Classification
	parser.add_argument('--image', type=str,
						help='The path to an image that will be classified. Classification only.')

	parser.add_argument('--classes', type=int,
						help='The number of classifications in the model. Classification only.')

	# Building Datasets
	parser.add_argument('--data_dir', type=str, default='/Data',
						help='Directory of original Dataset. Dataset Building only.')

	parser.add_argument('--label_dir', type=str, default='/Data',
						help='Directory of the labels used by some datasets. Dataset Building only.')

	parser.add_argument('--dataset', type=str,
						help='The dataset that is being processed. Dataset Building only.')

	parser.add_argument('--split_dir', type=str, default='none',
						help='The directory path to the split dataset. Dataset Building only.')

	parser.add_argument('--normalize', action='store_true', default=False,
						help='Should the dataset be normalized once completed. Dataset Building only.')

	# Run Parameters
	parser.add_argument('--address', type=str,
						help='The address to communicate with the MultiDS system.')

	return parser.parse_args()


def log(args, input_str, override=False):
	if args.verbose or override:
		print input_str
	if args.log != '':
		file = open(args.log, 'a')
		file.write(str(input_str) + '\n')
		file.close()


def main():
	args = parse_args()
	log(args, '\n' + str(args))
	if args.mode == 'train':
		train.train(args)
	elif args.mode == 'classify':
		classify(args)
	elif args.mode == 'dataset':
		build_dataset(args)
	elif args.mode == 'run':
		run_thread = run.MyThread(0, args)
		network_thread = run.MyThread(1, args)
		run_thread.start()
		network_thread.start()
		while network_thread.is_alive():
			continue
	else:
		log(args, 'Please select a mode using the tag --mode, use --help for help.', True)

if __name__ == '__main__':
	main()