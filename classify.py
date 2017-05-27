import np
import cv2
import main
import time
from lbpmapping import LBP
from classifier import Classifier


def classify(args):
	start_time = time.clock()

	lbp = LBP((256, 256))
	rgb_image = cv2.resize(cv2.imread(args.image, cv2.IMREAD_COLOR), (150, 150))
	lbp_image = lbp.map_lbp(rgb_image, 1, 8)
	frgb_image = 0  # Frontalization
	flbp_image = lbp.map_lbp(frgb_image, 1, 8)

	main.log(args, str(time.clock() - start_time) + ' Image Representations Extracted')

	classifications = []

	rgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, rgb_image.shape, colour=True, local=False)
	classifications.append(rgb_classifier.classify(rgb_image, (args.log + 'rgb')))

	lbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, lbp_image.shape, colour=True, local=False)
	classifications.append(lbp_classifier.classify(lbp_image, (args.log + 'lbp')))

	frgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, frgb_image.shape, colour=True, local=False)
	classifications.append(frgb_classifier.classify(frgb_image, (args.log + 'frgb')))

	lfrgb_classifier = Classifier(args, start_time, args.classes, args.resource_path, frgb_image.shape, colour=True, local=True)
	classifications.append(lfrgb_classifier.classify(frgb_image, (args.log + 'lfrgb')))

	flbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, flbp_image.shape, colour=True, local=False)
	classifications.append(flbp_classifier.classify(flbp_image, (args.log + 'flbp')))

	lflbp_classifier = Classifier(args, start_time, args.classes, args.resource_path, flbp_image.shape, colour=True, local=True)
	classifications.append(lflbp_classifier.classify(flbp_image, (args.log + 'lflbp')))

	result = np.zeros(args.classes)
	for classification in classifications:
		for i in range(len(classification)):
			result[i] += classification[i] / 6

	main.log(args, '\n' + str(time.clock() - start_time) + ' ' + args.image + ' classified as ' + str(result))
