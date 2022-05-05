import numpy as np
import math
import cv2


"""
Description:

Computes the homography matrix H between two pinhole camera views of a plane.
In particular, p2 = H * p1, where p1 is the pixel in the first view which
corresponds to a point p in the plane, and p2 is the pixel which corresponds
to p in the second view.

p    Point in the plane, in world coordinates
n    Plane normal, in world coordinates
K    Camera intrinsic matrix
R1   First pose rotation (world to camera)
t1   First pose translation, in world coordinates
R2   Second pose rotation (world to camera)
t2   Second pose translation, in world coordinates
"""
def computeHomography(p, n, K, R1, t1, R2, t2):
	m = n.T @ (p - t1)
	H = K @ R2 @ (m[0] @ np.eye(3) + (t1 - t2) @ n.T) @ R1.T @ np.linalg.inv(K)

	return H

"""
Description
"""
def buildSquare(center, size):
	center = center.astype(int)
	rowtop = center[0] - size/2
	coltop = center[1] - size/2
	# rowbot = center[0] + size/2 + 1
	# colbot = center[1] + size/2 + 1

	return np.array([rowtop, coltop, size, size]).astype(int)


"""
Description
"""
def removeIndicesLst(vec, indices):

    for i in range(len(indices)-1, -1, -1):
    	vec.pop(indices[i])
        
    return vec 

"""
Description
"""
def removeIndicesArr(vec, indices):

    vec = np.delete(vec, indices, axis=0)

    return vec    

# """
# Description
# """
# def removeRowCol(mat, idx):

#     int n = mat.rows;

#     Mat reduced(n - 1, n - 1, mat.type());

#     if (idx == 0)
#         mat(Rect(1, 1, n - 1, n - 1)).copyTo(reduced(Rect(0, 0, n - 1, n - 1)));

#     else if (idx == n - 1)
#         mat(Rect(0, 0, n - 1, n - 1)).copyTo(reduced(Rect(0, 0, n - 1, n - 1)));

#     else {
#         mat(Rect(0, 0, idx, idx)).copyTo(reduced(Rect(0, 0, idx, idx)));
#         mat(Rect(idx+1, 0, n-idx-1, idx)).copyTo(reduced(Rect(idx, 0, n-idx-1, idx)));
#         mat(Rect(0, idx+1, idx, n-idx-1)).copyTo(reduced(Rect(0, idx, idx, n-idx-1)));
#         mat(Rect(idx+1, idx+1, n-idx-1, n-idx-1)).copyTo(reduced(Rect(idx, idx, n-idx-1, n-idx-1)));
#     }

#     return reduced

# def removeRowCol(mat, idx):
# 	return reduced


# """
# Description
# """
# def removeRowsCols():
# 	pass


"""
Description
"""
def removeRow():
	pass


# """
# Description
# """
# def removeRows():
# 	pass


"""
Description
"""
def computeEllipses(center, S):
	ellipses = np.empty((0, 5))
	for i in range(center.shape[0]):
		#Compute Eigen values and Eigen vectors
		Si = S[2*i : 2*(i + 1), 2*i : 2*(i + 1)]
		ret, eigenval, eigenvec = cv2.eigen(Si)

		#Compute angle
		angle = math.atan2(eigenvec[0, 1], eigenvec[0, 0])

		#Shift angle
		if (angle < 0):
			angle += math.pi
		angle = math.degrees(angle)

		majorAxis = 4 * math.sqrt(eigenval[0])
		minorAxis = 4 * math.sqrt(eigenval[1])

		ellipses = np.vstack((ellipses, np.array([center[i][0], center[i][0], majorAxis, minorAxis, -angle])))
	
	return ellipses


"""
Description
"""
def getBoundingBox(ellipse, imageSize):

    angle = math.radians(ellipse[4])
    x = ellipse[0]
    y = ellipse[1]
    a = 0.5 * ellipse[2]
    b = 0.5 * ellipse[3]

    if angle == 0 or a == b:

        xmax = a
        ymax = b

    else:
        a2inv = 1 / (a*a)
        b2inv = 1 / (b*b)

        c = math.cos(angle)
        s = math.sin(angle)
        c2 = c*c
        s2 = s*s

        A = c2 * a2inv + s2 * b2inv
        B = c2 * b2inv + s2 * a2inv
        C = 2 * c * s * (b2inv - a2inv)

        r = (4 * A * B / (C * C) - 1)
        x0 = 1 / math.sqrt(A * r)
        y0 = 1 / math.sqrt(B * r)

        if C < 0:
            x0 = -x0
            y0 = -y0

        xmax = 2 * B * y0 / C
        ymax = 2 * A * x0 / C

    x0 = x - xmax
    y0 = y - ymax
    w = 2 * xmax + 1
    h = 2 * ymax + 1

    if (x0 + w) > imageSize[1]:
        w = imageSize[1] - x0
    if (y0 + h) > imageSize[0]:
        h = imageSize[0] - y0

    return np.array([max(y - ymax, 0.), max(x - xmax, 0.), h, w])


# """
# Description
# """
# def drawRectangle():
# 	pass



"""
Description
"""
def drawSquare(image, center, width, color):
	points = buildSquare(center, width)
	image = cv2.rectangle(image, tuple(points[0:2]), tuple(points[0:2]+points[2:4]), 0, 2, cv2.LINE_AA)
	return image

"""
Description
"""
def drawEllipse(image, e, color):

    image = cv2.ellipse(image, e[0:2], e[2:4], e[4], color, 1, cv2.LINE_AA)
    return image


# """
# Description
# """
# def drawCircle():
# 	pass


"""
Description
"""
def drawTemplate(image, templ, position, cx, cy):
    templ_x = 0
    image_x = position[0] - cx

    if image_x > (image.shape[1] - 1):
        return

    if image_x < 0:
        templ_x = - image_x
        image_x = 0

    templ_y = 0
    image_y = position[1] - cy

    if image_y > image.shape[0] - 1:
        return

    if image_y < 0:
        templ_y = - image_y
        image_y = 0

    w = templ.shape[1] - templ_x;

    if w <= 0:
        return

    if image_x + w > image.shape[1]:
        w = image.shape[1] - image_x
        h = templ.shape[0] - templ_y

    if h <= 0:
        return

    if image_y + h > image.shape[0]:
        h = image.shape[0] - image_y

    image[image_y:h, image_x:w] = templ[templ_y:h, templ_x:w]

    return image
