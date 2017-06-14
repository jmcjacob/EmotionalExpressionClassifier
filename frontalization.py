import face_specific_augm.check_resources as check
import face_specific_augm.ThreeD_Model as ThreeD_Model
import face_specific_augm.camera_calibration as camera
import face_specific_augm.facial_feature_detector as features
from face_specific_augm.renderer import render


class Front:
	def __init__(self, args):
		self.args = args
		check.check_dlib_landmark_weights(args)
		self.model3d = ThreeD_Model.FaceModel(args.resource_path + 'model3Ddlib.mat', 'model3D') # This is wrong and needs to be changed!
		self.eyemask = self.model3d.eyemask

	def frontalized(self, image):
		landmarks = features.get_landmarks(image, self.args.resource_path, self.args)
		if len(landmarks) > 0:
			proj_matrix, camera_matrix, rmat, tvec = camera.estimate_camera(self.model3d, landmarks[0])
			_, front_image, _, _, _, _ = render(image, proj_matrix, self.model3d.ref_U, self.eyemask, self.model3d.facemask, True)
			return front_image
