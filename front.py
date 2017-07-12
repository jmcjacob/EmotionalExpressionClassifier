import sys
import dlib
import numpy as np
import scipy.io as io
import face_specific_augm.check_resources as check
sys.path.insert(0, 'face-frontalization')
import camera_calibration as camera
import facial_feature_detector as facial
import frontalize as front


class Front:
	def __init__(self, args):
		self.args = args
		check.check_dlib_landmark_weights(args)
		self.detector = dlib.get_frontal_face_detector()
		self.model3D = front.ThreeD_Model(args.resource_dir + '/Image/model3Ddlib.mat', 'model_dlib')
		self.eyemask = np.asarray(io.loadmat(args.resource_dir + '/Image/eyemask.mat')['eyemask'])

	def frontalized(self, image):
		landmarks = facial.get_landmarks(image, self.args.resource_dir, self.args)
		if len(landmarks) > 0:
			proj_matrix, camera_matrix, rmat, tvec = camera.estimate_camera(self.model3D, landmarks[0])
			_, front_image = front.frontalize(image, proj_matrix, self.model3D.ref_U, self.eyemask)
			detection = self.detector(front_image, 1)
			for _, detection in enumerate(detection):
				return front_image[detection.top():detection.bottom(), detection.left():detection.right()]
			return