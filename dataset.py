import os
import cv2
import csv
import dlib
import main
import time
import front as frontalization
import random
import shutil
import numpy as np
import skvideo.io
#import frontalization
#from skimage import feature


def build_dataset(args):
	start_time = time.clock()
	if args.dataset == 'CK+':
		build_structure(args, start_time, 8)
		image_files = []
		for outer_folder in os.listdir(args.data_dir):
			if os.path.isdir(args.data_dir):
				for inner_folder in os.listdir(args.data_dir + '/' + outer_folder):
					if os.path.isdir(args.data_dir + '/' + outer_folder + '/' + inner_folder):
						for input_file in os.listdir(args.data_dir + '/' + outer_folder + '/' + inner_folder):
							data_type = input_file.split('.')[1].lower()
							if not(data_type == 'png' or data_type == 'jpg' or data_type == 'tiff'):
								break
							label_file = args.label_dir + '/' + outer_folder + '/' + inner_folder + '/' + input_file[:-4] + '_emotion.txt'
							if os.path.isfile(label_file):
								read_file = open(label_file, 'r')
								label = int(float(read_file.readline()))
								for i in range(-1, -6, -1):
									image_file = sorted(os.listdir(args.data_dir + '/' + outer_folder + '/' + inner_folder))[i]
									data_type = image_file.split('.')[1].lower()
									if (data_type == 'png' or data_type == 'jpg' or data_type == 'tiff'):
										image_files.append((args.data_dir + '/' + outer_folder + '/' + inner_folder + '/' + image_file, label))
								neutral_file = sorted(os.listdir(args.data_dir + '/' + outer_folder + '/' + inner_folder))[0]
								data_type = neutral_file.split('.')[1]
								if not(data_type == 'png' or data_type == 'jpg' or data_type == 'tiff'):
									neutral_file = sorted(os.listdir(args.data_dir + '/' + outer_folder + '/' + inner_folder))[1]
								image_files.append((args.data_dir + '/' + outer_folder + '/' + inner_folder + '/' + neutral_file, 0))
		main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(image_files)) + ' Images File Collected')
		extract_images(args, start_time, image_files)

	elif args.dataset == 'KDEF':
		build_structure(args, start_time, 7)
		image_files = []
		for folder in os.listdir(args.data_dir):
			if os.path.isdir(args.data_dir + '/' + folder):
				for file in os.listdir(args.data_dir + '/' + folder):
					data_type = file.split('.')[1].lower()
					if (data_type == 'png' or data_type == 'jpg' or data_type == 'tiff') and file[6] != 'F':
						label = 0
						if file[4:6] == 'AF':
							label = 3
						elif file[4:6] == 'AN':
							label = 1
						elif file[4:6] == 'DI':
							label = 2
						elif file[4:6] == 'HA':
							label = 4
						elif file[4:6] == 'NE':
							label = 0
						elif file[4:6] == 'SA':
							label = 5
						elif file[4:6] == 'SU':
							label = 6
						image_files.append((args.data_dir + '/' + folder + '/' + file, label))
		main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(image_files)) + ' Images File Collected')
		extract_images(args, start_time, image_files)

	elif args.dataset == 'AMFED':
		build_structure(args, start_time, 3)
		images = []
		for video in os.listdir(args.data_dir):
			if video.split('.')[1] == 'avi':
				filename = args.data_dir + video
				#print filename
				try:
					cap = skvideo.io.vread(filename)
				except:
					print 'Error with ' + filename
				with open(args.label_dir + '/' + video[:-4] + '-label.csv', 'rU') as csvfile:
					reader = csv.reader(csvfile, delimiter=',', dialect=csv.excel_tab)
					skip = True
					for row in reader:
						if skip:
							skip = False
							continue
						if float(row[1]) > 33:
							frame_no = float(row[0]) * 14
							images.append((cap[int(frame_no)], 2))
						elif float(row[3]) > 33:
							frame_no = float(row[0]) * 14
							images.append((cap[int(frame_no)], 1))
						else:
							frame_no = float(row[0]) * 14
							images.append((cap[int(frame_no)], 0))
		main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(len(images)) + ' Images File Collected')
		extract_images(args, start_time, images, False)

	else:
		main.log(args, 'Please specify a dataset \'--dataset\'', True)
	if args.split_dir != 'none':
		splits = split_images(args, start_time)
		main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + 'Images have been split ')
	if args.normalize:
		normalize(args, start_time)
		if args.split_dir != 'none':
			normalize(args, start_time, dirs=splits)


def build_structure(args, start_time, classes):
	for folder in [args.training_dir, args.testing_dir]:
		if not os.path.exists(folder):
			os.makedirs(folder)
			os.makedirs(folder + '/rgb')
			for i in range(classes):
				os.makedirs(folder + '/rgb/' + str(i))
			# os.makedirs(folder + '/lbp')
			# for i in range(classes):
			# 	os.makedirs(folder + '/lbp/' + str(i))
			os.makedirs(folder + '/frgb')
			for i in range(classes):
				os.makedirs(folder + '/frgb/' + str(i))
			# os.makedirs(folder + '/flbp')
			# for i in range(classes):
			# 	os.makedirs(folder + '/flbp/' + str(i))
	main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's Folder Structure Built')


def extract_images(args, start_time, image_files, images=True):
	test_length = int(round(0.2 * len(image_files)))
	shuffled = image_files[:]
	random.shuffle(shuffled)
	training_data = shuffled[test_length:]
	testing_data = shuffled[:test_length]
	save_image(args, start_time, args.training_dir, training_data, 'training', images)
	save_image(args, start_time, args.testing_dir, testing_data, 'testing', images)


def save_image(args, start_time, save, data, type, images=True):
	detector, count = dlib.get_frontal_face_detector(), 0
	front = frontalization.Front(args)
	for image_file in data:
		if images:
			image = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
		else:
			image = image_file[0]
		detections = detector(image, 1)
		for _, detection in enumerate(detections):
			#try:
			face = image[detection.top():detection.bottom(), detection.left():detection.right()]
			if face.shape[0] < 10 or face.shape[1] < 10:
				continue
			face = cv2.resize(face, (88, 88))
			images = []
			images.append(face)
			images.append(hue(face, 5))
			images.append(hue(face, -5))
			images.append(noisy('sp', images[0]))
			images.append(noisy('gauss', images[0]))
			images.append(hue(noisy('sp', images[0]), 5))
			images.append(hue(noisy('sp', images[0]), -5))
			images.append(hue(noisy('gauss', images[0]), 5))
			images.append(hue(noisy('gauss', images[0]), -5))
			for _image in images:
				if images:
					cv2.imwrite(save + '/rgb/' + str(image_file[1]) + '/' + str(count) + '.jpg', _image)
				else:
					cv2.imwrite(save + '/rgb/' + str(image_file[1]) + '/' + str(count) + '.jpg', _image)
				#lbp_image = feature.local_binary_pattern(cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform').astype(np.uint8)
				#lbp_image *= (255 / lbp_image.max())
				#cv2.imwrite(save + '/lbp/' + str(image_file[1]) + '/' + str(count + 1) + '.jpg', lbp_image)
				frgb_image = front.frontalized(image)
				if frgb_image is None:
					continue
				if frgb_image.shape[0] < 10 or frgb_image.shape[1] < 10:
					continue
				cv2.imwrite(save + '/frgb/' + str(image_file[1]) + '/' + str(count + 2) + '.jpg', cv2.resize(frgb_image, (88,88)))
				#flbp_image = feature.local_binary_pattern(cv2.cvtColor(frgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform').astype(np.uint8)
				#flbp_image *= (255 / flbp_image.max())
				#cv2.imwrite(save + '/flbp/' + str(image_file[1]) + '/' + str(count + 3) + '.jpg', flbp_image)
				count += 4
				if count % 100 == 0:
					main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + str(count) + ' ' + type +' images extracted')
			#except:
				#pass
	main.log(args, str(time.clock() - start_time) + ' ' + type + ' Images Extracted')


def split_images(args, start_time):
	output_dirs = []
	for i in [args.training_dir, args.testing_dir]:
		for j in ['rgb', 'frgb']:
			input_dir = str(i) + '/' + str(j)
			output_dir = args.split_dir + '/' + i.split('/')[-1] + '/' + j
			output_dirs.append(output_dir)
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
				for m in range(1, 3):
					os.makedirs(output_dir + '/' + str(m))
			shutil.copytree(input_dir + '/0', output_dir + '/0')
			if args.dataset == 'CK+':
				negative = [1, 3, 4, 6]
				positive = [2, 5, 7]
			elif args.dataset == 'KDEF':
				negative = [1, 2, 3, 5]
				positive = [4, 6]
			for n in negative:
				for image_file in os.listdir(input_dir + '/' + str(n)):
					shutil.copy(input_dir + '/' + str(n) + '/' + image_file, output_dir + '/' + str(1) + '/' + image_file)
			for n in positive:
				for image_file in os.listdir(input_dir + '/' + str(n)):
					shutil.copy(input_dir + '/' + str(n) + '/' + image_file, output_dir + '/' + str(2) + '/' + image_file)
	return output_dirs


def normalize(args, start_time, dirs=[]):
	if len(dirs) == 0:
		for i in [args.training_dir, args.testing_dir]:
			for j in ['rgb/','frgb/']:
				dirs.append(i + '/' + j)
	for dir in dirs:
		minimum, num_files = 1000000, []
		for folder in os.listdir(dir):
			path = dir + '/' + folder
			files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
			num_files.append([files, path])
			main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's ' + path + ' has ' + str(files) + ' files')
			if files < minimum:
				minimum = files
		main.log(args, '{:.5f}'.format(time.clock() - start_time) + 's minimum = ' + str(minimum))

		for i in range(len(num_files)):
			while num_files[i][0] > minimum:
				os.remove(num_files[i][1] + '/' + random.choice(os.listdir(num_files[i][1])))
				num_files[i][0] -= 1



def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy.astype(np.uint8)
	elif noise_typ == "sp":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
		out[coords] = 1
		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out


def hue(image, value):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if not image[i][j][0] == 0 and not image[i][j][1] == 0:
				temp = int(image[i][j][0]) + value
				if not temp >= 255 or not temp <= 0:
					image[i][j][0] = image[i][j][0] + value
	return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
