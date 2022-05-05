import numpy as np
import cv2

class FeatureNew:
	def __init__(self, image, roi, normal, R, t):
		
		self.image = image
		self.roi = roi
		self.pos = None
		self.normal = normal
		self.R = R
		self.t = t

		self.P = None

		self.matchingFails = 0
		self.matchingAttempts = 1

		self.dir = None
		self.depths = None
		self.probs = None


class Feature:
	def __init__(self, image, roi, R, t, dir_, depthInterval, depthSamples):
		
		self.image = image
		self.roi = roi
		self.pos = None
		self.R = R
		self.t = t
		self.dir = dir_

		self.depths = np.zeros(depthSamples)
		self.probs = np.ones(depthSamples)/depthSamples

		self.P = None

		self.matchingFails = 0
		self.matchingAttempts = 1

		self.dir = None
		