import cv2
import dlib
import main
import time
import numpy as np
import frontalization
from skimage import feature
from classifier import Classifier


def classify(args):
	start_time = time.clock()
	detector = dlib.get_frontal_face_detector()
	front = frontalization.Front(args)
	image = cv2.resize(cv2.imread(args.image, cv2.IMREAD_COLOR), (150, 150))
	detection = detector(image, 1)
	for _, detection in enumerate(detection):
		rgb_image = image[detection.top():detection.bottom(), detection.left():detection.right()]
		lbp_image = feature.local_binary_pattern(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform')
		frgb_image = front.frontalized(image)
		flbp_image = feature.local_binary_pattern(cv2.cvtColor(frgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform')

		main.log(args, str(time.clock() - start_time) + ' Image Representations Extracted')

		classifications = []

		rgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, rgb_image.shape, colour=True, local=False)
		classifications.append(rgb_classifier.classify(rgb_image, (args.log + 'rgb')))

		lbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, lbp_image.shape, colour=False, local=False)
		classifications.append(lbp_classifier.classify(lbp_image, (args.log + 'lbp')))

		frgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, frgb_image.shape, colour=True, local=False)
		classifications.append(frgb_classifier.classify(frgb_image, (args.log + 'frgb')))

		lfrgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, frgb_image.shape, colour=True, local=True)
		classifications.append(lfrgb_classifier.classify(frgb_image, (args.log + 'lfrgb')))

		flbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, flbp_image.shape, colour=False, local=False)
		classifications.append(flbp_classifier.classify(flbp_image, (args.log + 'flbp')))

		lflbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, flbp_image.shape, colour=False, local=True)
		classifications.append(lflbp_classifier.classify(flbp_image, (args.log + 'lflbp')))

		result = np.zeros(args.classes)
		for classification in classifications:
			for i in range(len(classification)):
				result[i] += classification[i] / 6

		main.log(args, '\n' + str(time.clock() - start_time) + ' ' + args.image + ' classified as ' + str(result), True)
		return True
	main.log(args, '\n' + str(time.clock() - start_time) + ' Could not detect face in inputted image', True)
