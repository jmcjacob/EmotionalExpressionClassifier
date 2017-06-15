import cv2
import sys
import dlib
import main
import time
import select
import threading
import websocket
import numpy as np
import frontalization
from skimage import feature
from classifier import Classifier


class MyThread(threading.Thread):
	def __init__(self, id, args):
		threading.Thread.__init__(self)
		self.id = id
		self.args = args

	def run(self):
		if self.id == 0:
			main.log(self.args, 'Starting ' + str(self.id))
			run(self.args)
			main.log(self.args, 'Exiting ' + str(self.id))
		elif self.id == 1:
			main.log(self.args, 'Starting ' + str(self.id))
			run_network(self.args)
			main.log(self.args, 'Exiting ' + str(self.id))


def run(args):
	global averages, running
	averages, running = True, []
	start_time = time.clock()
	video = cv2.VideoCapture()
	q = []

	detector = dlib.get_frontal_face_detector()
	front = frontalization.Front(args)

	classifiers = []
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150,150,3), colour=True, local=False))
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150, 150), colour=False, local=False))
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150,150,3), colour=True, local=False))
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150,150,3), colour=True, local=True))
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150, 150), colour=False, local=False))
	classifiers.append(Classifier(args, start_time, args.classes, args.resource_dir, (150, 150), colour=False, local=True))

	if video.grab():
		while running:
			_, frame = video.read()
			image = cv2.resize(frame, (150, 150, frame.shape[2]))
			detection = detector(image, 1)
			if detection:
				rgb_image = frame[detection.top():detection.bottom(), detection.left():detection.right()]
				lbp_image = feature.local_binary_pattern(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform')
				frgb_image = front.frontalized(image)
				flbp_image = feature.local_binary_pattern(cv2.cvtColor(frgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64), 8, 1, 'uniform')

				classifications = []
				classifiers[0].classify(rgb_image, args.log + 'rgb')
				classifiers[1].classify(lbp_image, (args.log + 'lbp'))
				classifiers[2].classify(frgb_image, (args.log + 'frgb'))
				classifiers[3].classify(frgb_image, (args.log + 'lfrgb'))
				classifiers[4].classify(flbp_image, (args.log + 'flbp'))
				classifiers[5].classify(flbp_image, (args.log + 'lflbp'))

				result = np.zeros(args.classes)
				for classification in classifications:
					for i in range(len(classification)):
						result[i] += classification[i] / 6

				temp = []
				if len(q) < 10:
					q.insert(0, result)
				else:
					q.pop()
					q.insert(0, result)
				for i in range(len(q[0])):
					average = 0
					for j in range(len(q)):
						average += q[j][i] / 10
					temp.append(average)
				averages = temp
				main.log(args, averages)
			else:
				main.log(args, 'No Face Found', True)
			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				raw_input()
				break


def on_open(ws):
	ws.send('EMOTION')
	print 'Connected to server!'


def on_close(ws):
	print 'Connection Closed'


def on_error(ws, err):
	print err.message


def on_data(ws, data):
	global averages
	print 'Recived: ' + data
	if data == 'REQUEST':
		string = ''
		for i in averages:
			string += str(i) + ' '
		print 'Sent: ' + string
		ws.send(string.replace('[', '').replace(']', ''))
		averages = []
	elif data == 'CLOSE':
		ws.close()


def run_network(args):
	global running
	try:
		ws = websocket.WebSocketApp(args.address, on_message=on_data, on_error=on_error, on_close=on_close)
		ws.on_open = on_open
		ws.run_forever()
		running = False
	except Exception as err:
		main.log(args, err.message, True)
