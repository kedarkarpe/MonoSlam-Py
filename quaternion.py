import numpy as np
import math


"""
Description:

Builds a rotation quaternion from a (non-zero) rotation vector.
"""
def computeQuaternion(axis):
	q = np.zeros(shape=(4, ))
	angle = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)

	q[0] = math.cos(angle/2)
	q[1] = axis[0] * math.sin(angle/2) / angle 
	q[2] = axis[1] * math.sin(angle/2) / angle
	q[3] = axis[2] * math.sin(angle/2) / angle

	return q

"""
Description:

Computes Hamilton product of two quaternions.
"""
def quaternionMultiply(p, q):
	t0 = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
	t1 = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
	t2 = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
	t3 = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]

	product = np.array([t0, t1, t2, t3])

	return product

"""
Description:

Computes a rotation matrix from a rotation quaternion. In particular, if the
quaternion is an orientation quaternion, which rotates a reference frame A to
its final pose B, the matrix is a change of basis from B to A.
"""
def getRotationMatrix(q):
	R = np.array([[q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2]-2*q[0]*q[3],     2*q[1]*q[3]+2*q[0]*q[2]],
		[2*q[1]*q[2]+2*q[0]*q[3],     q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], 2*q[2]*q[3]-2*q[0]*q[1]],
		[2*q[1]*q[3]-2*q[0]*q[2],     2*q[2]*q[3]+2*q[0]*q[1],     q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]])

	return R


"""
Description:

Computes Inverse of Quaternion
"""
def quaternionInv(q):
  q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
  return q_inv

"""
Description:

Computes the matrices of the derivatives of R(q)^t with respect to the quaternion
components. That is, the entries of Ri are the derivatives of the entries of the
transpose of the matrix returned by getRotationMatrix(q) with respect to the ith
quaternion component.
"""
def getRotationMatrixDerivatives(q):

  t0, t1, t2, t3 = q[0], q[1], q[2], q[3]

  R1 = 2 * np.array([t0, t3, -t2, -t3, t0, t1, t2, -t2, t0]).reshape(3, 3)
  R2 = 2 * np.array([t1, t2, t3, t2, -t1, t0, t3, -t0, -t1]).reshape(3, 3)
  R3 = 2 * np.array([-t2, t1, -t0, t1, t2, t3, t0, t3, -t2]).reshape(3, 3)
  R4 = 2 * np.array([-t3, t0, t1, -t0, -t3, t2, t1, t2, t3]).reshape(3, 3)

  return R1, R2, R3, R4
