import cv2
import numpy as np
import os
import glob
import time
import yaml


def calibration(images, dims):

    CHECKERBOARD = dims

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    threedpoints = []
    twodpoints = []
 
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    prev_img_shape = None

    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            threedpoints.append(objectp3d)
 
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            print(corners.shape, corners2.shape)
            twodpoints.append(corners2)
 
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
    
        cv2.imshow('img', image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = image.shape[:2]

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    data = {"Image_Size": [h, w], "Intrinsic_Matrix": matrix.tolist(), "Distortion_Coefficients": distortion.tolist()}
    fname = "intrinsics.xml"

    with open(fname, "w") as f:
        yaml.dump(data, f)


    print(" Camera matrix:")
    print(matrix)
 
    print("\n Distortion coefficient:")
    print(distortion)
     
    print("\n Rotation Vectors:")
    print(r_vecs)
     
    print("\n Translation Vectors:")
    print(t_vecs)

    return matrix, distortion, r_vecs, t_vecs

if __name__ == '__main__':
    calibration_images = glob.glob('img/*.jpg')
    checkerboard_dims = (7, 9)

    print('\nStart Intrinsic Parameters Calculation!')
    K, _, R, t = calibration(calibration_images, checkerboard_dims)
