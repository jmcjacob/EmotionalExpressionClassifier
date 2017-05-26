import cv2
import math
import numpy as np
from bitsets import bitset
from sklearn.manifold import MDS


class LBP:
	def __init__(self, input_shape):
		self.cyclic_map = self.mds(self.cyclic_mapping(input_shape))
		self.regular_map = self.mds(self.regular_mapping(input_shape))
		self.lbp_mapping = self.get_mapping(8)

	def map_lbp(self, image, radius, neighbors=0, mode='reg'):
		h1 = self.get_lbp(image, radius, neighbors, self.lbp_mapping)
		if mode == 'reg':
			mds_im = self.regular_map[h1, :]
		else:
			mds_im = self.cyclic_map[h1, :]
		out_im = np.reshape(mds_im, [len(h1), 3])
		return out_im

	@staticmethod
	def get_lbp(image, radius, neighbors, mapping):
		image = cv2.colorChange(image, cv2.COLOR_BGR2GRAY).astype(np.double)
		spoints = np.zeros(neighbors, 2)
		a = 2 * math.pi / neighbors
		for i in range(neighbors):
			spoints[i, 0] = -radius*math.sin(i * a)
			spoints[i, 1] = radius*math.cos(i * a)

		xsize, ysize = image.shape
		miny = min(spoints[:, 1])
		maxy = max(spoints[:, 1])
		minx = min(spoints[:, 2])
		maxx = max(spoints[:, 2])

		bsizey = math.ceil(max(maxy, 0)) - math.floor(min(miny, 0))
		bsizex = math.ceil(max(maxx, 0)) - math.floor(min(minx, 0))
		origy = 1 - math.floor(min(miny, 0))
		origx = 1 - math.floor(min(minx, 0))

		dx, dy = xsize - bsizex, ysize - bsizey
		C = image[origy : origy + dy, origx : origx + dx].astype(np.double)
		bins = 2 ** neighbors
		result = np.zeros((int(dx), int(dy)))

		for i in range(neighbors):
			y, x = spoints[i, 0] + origy, spoints[i, 1] + origx
			fy, fx = math.floor(y), math.floor(x)
			cy, cx = math.ceil(y), math.ceil(x)
			ry, rx = round(y), round(x)
			if abs(x - rx) < 1e-6 and abs(y -ry) < 1e-6:
				N = image[ry:ry+dy, rx:rx+dx]
				D = N >= C
			else:
				ty = y - fy
				tx = x - fx
				w1 = (1 - tx) * (1 - ty)
				w2 = tx * (1 - ty)
				w3 = (1 - tx) * ty
				w4 = tx * ty
				N = w1 * image[fy:fy+dy, fx:fx+dx] + w2 * image[fy:fy+dy, cx:cx+dx] + w3 * image[fy:fy+dy, cx:cx+dx] + w4 * image[fy:fy+dy, cx:cx+dx]
				D = N >= C
			v = 2 ^ (i - 1)
			result = result + v * D

		if mapping != 0:
			bins = mapping[2]
			for i in range(result[0]):
				for j in range(result[1]):
					result[i, j] = mapping[0][result[i,j]]

		return result.astype(np.uint8)

	@staticmethod
	def get_mapping(samples):
		table = range(2 ** samples - 1)
		new_max, index = samples + 2, 0
		for i in range(2 ** samples - 1):
			j = bitset(i >> 1, 1, ((samples&(1<<1))!=0))
			numt = sum((((i|j)&(1<<1))!=0), range(samples))
			if numt <= 2:
				table[i] = sum(((range(samples)&(1<<i))!=0))
			else:
				table[i] = samples
		return (table, samples, new_max)

	def cyclic_mapping(self, input_shape):
		cyclic_dist_matrix = np.zeros(input_shape)
		for ix in range(input_shape[0]):
			ix_bin_vec = [0, self.de2bi(ix, 8)]
			for jx in range(input_shape[1]):
				jx_bin_vec = [0, self.de2bi(jx, 8)]
				dist1 = self.distEmd(ix_bin_vec, jx_bin_vec)
				dist2 = self.distEmd(reversed(ix_bin_vec), jx_bin_vec)
				dist3 = self.distEmd(ix_bin_vec, reversed(jx_bin_vec))
				cyclic_dist_matrix[ix, jx] = min([dist1, dist2, dist3])
		return cyclic_dist_matrix

	def regular_mapping(self, input_shape):
		dist_matrix = np.zeros(input_shape)
		for ix in range(255):
			ix_bin_vec = self.de2bi(ix, 8)
			for jx in range(255):
				jx_bin_vec = self.de2bi(jx, 8)
				dist_matrix[ix, jx] = dist_matrix(ix_bin_vec, jx_bin_vec)
		return dist_matrix

	def mds(self, matrix):
		y = MDS(3).fit(matrix)
		y2 = y + abs(min(y[:]))
		y2 = y2 * (255 / max(y2[:]))
		return np.uint8(y2)

	@staticmethod
	def de2bi(input, size):
		output = list(bin(input)[2:])
		while len(output) != size:
			output.insert(0, 0)
		return output

	@staticmethod
	def distEmd(x, y):
		xcdf, ycdf = np.cumsum(x, 2), np.cumsum(y, 2)
		m, n = len(x[0]), len(y[0])
		m_ones, d = np.ones((1, m)), np.zeros((n, m))
		for i in range(n):
			tycdf = ycdf[i,:]
			ycdf_rep = tycdf[m_ones, :]
			d[:, i] = sum(abs(xcdf - ycdf_rep), 2)
		return d
