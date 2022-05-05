import numpy as np
import cv2

from utils import *

class camera:
	def __init__(self, K, distCoeffs, frameSize):
		
		self.r = np.empty((1, 3))
		self.q = np.empty((1, 3))
		self.v = np.empty((1, 3))
		self.w = np.empty((1, 3))

		self.P = np.empty((3, 3))

		self.R = np.empty((3, 3))
		self.K = K
		self.distCoeffs = distCoeffs
		self.frameSize = frameSize

	"""
	Description
	"""
	def warpPatch(self, p, n, view1, patch1, R1, t1, R2, t2):
		
		# Create black background
		black = np.zeros(view1.shape)

		# Copy the original patch into the black background
		black[patch1[0]:patch1[2], patch1[1]:patch1[3]] = view1[patch1[0]:patch1[2], patch1[1]:patch1[3]]

		# Compute the homography matrix between the first and second views
		H = computeHomography(p, n, self.K, R1, t1, R2, t2)

		# Apply the homography
		warped = cv2.warpPerspective(black, H, view1.shape)

		# Set every non-black pixel of the warped image to white
		ret, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)

		# Use the binary mask to crop the warped image so that it only contains the patch
		points = np.argwhere(binary > 0)
		patchBox = np.array(cv2.boundingRect(points))

		# Compute the projection of the center of the patch
		x = int(patch1[1] + patch1[3] / 2)
		y = int(patch1[0] + patch1[2] / 2)

		centerProj = H @ np.array([x, y, 1])
		centerProj = centerProj/ centerProj[2]

		u = centerProj[1] - patchBox[1].x
		v = centerProj[0] - patchBox[0].y

		return warped[patchBox[0]:patchBox[2], patchBox[1]:patchBox[3]], u, v


	"""
	Description
	"""
	def projectPoints(self, R, t, points3D):
		points2D, _ = cv2.projectPoints(points3D, R, t, self.K, self.distCoeffs)
		return points2D