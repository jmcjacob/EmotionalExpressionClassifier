import cv2
import dlib
import main
import time
import threading
import websocket
import numpy as np
import frontalization
from naoqi import ALProxy
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
	detector = dlib.get_frontal_face_detector()
	front = frontalization.Front(args)
	global predictions, running
	predictions, running = [], True
	start_time = time.clock()
	q = []

	videoDevice = ALProxy('ALVideoDevice', args.ip, args.port)
	captureDevice = videoDevice.subscribeCamera('Camera', 0, 2, 13, 10)
	image = np.zeros((480, 640, 3), np.uint8)

	classifier = Classifier(args, start_time, 3, args.resource_dir, [88, 88, 3], args.scope, '/'+args.model+'/')
	classifier.load_model()

	while running:
		image = stream2cv(image, videoDevice.getImageRemote(captureDevice))
		if not image:
			detection = detector(image, 1)
			if detection:
				rgb_image = image[detection.top():detection.bottom(), detection.left():detection.right()]
				rgb_image = cv2.resize(rgb_image, (88, 88))
				front_image = front.frontalized(rgb_image)
				prediction = classifier.classify(front_image)
				predictions.append(prediction)
				main.log(args, '\n{:.5f}s'.format(time.clock() - start_time) + ' Prediction: ' + str(prediction))
				main.log(args, '\n{:.5f}s'.format(time.clock() - start_time) + ' Averages: ' + str(predictions))


def on_open(ws):
	ws.send('EMOTION')
	print 'Connected to server!'


def on_close(ws):
	print 'Connection Closed'


def on_error(ws, err):
	print err.message


def on_data(ws, data):
	global predictions
	print 'Recived: ' + data
	if data == 'REQUEST':
		string = ''
		for i in predictions:
			string += str(i) + ' '
		print 'Sent: ' + string
		ws.send(string.replace('[', '').replace(']', ''))
		predictions = []
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


def stream2cv(image, stream):
	if stream[6] is None:
		return False
	values = map(ord, list(stream))
	i = 0
	for y in range(0, image.shape[0]):
		for x in range(0, image.shape[1]):
			image.itemset((y, x, 0), values[i + 0])
			image.itemset((y, x, 1), values[i + 1])
			image.itemset((y, x, 2), values[i + 2])
			i += 3
	return image
