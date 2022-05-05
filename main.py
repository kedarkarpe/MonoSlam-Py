import time
import yaml

import numpy as np
import cv2

# import camera
# import utils
from mapmanager import Map
# import quaternion
# import feature

def readFrame(cap):

    tol = 1 / 30
    elapsed = tol

    while elapsed < tol:
        t0 = time.time()
        cap.grab()
        elapsed = time.time() - t0

    ret, frame = cap.read()

    return frame


def main():


    """ 
    Load Intrinsics and Extrinsics from calibration file 
    """
    calibrationFile = 'calibration/intrinsics.yaml'

    with open(calibrationFile) as file:
        try:
            data = yaml.safe_load(file)   

        except yaml.YAMLError as exc:
            print(exc)


    frameSize   = np.array(data["Image_Size"])
    K           = np.array(data["Intrinsic_Matrix"])
    distCoeffs  = np.array(data["Distortion_Coefficients"])

    patternRows, patternCols = 8, 8
    squareSize = 4.2


    """ 
    Meta parameters 
    """
    patchSize = 22
    minDensity = 8
    maxDensity = 18
    failTolerance = 0.5

    # Checkerboard Params
    patternSize = (7, 9)
    squareSize = 0.02
    var = np.array([0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.008, 0.008, 0.008])

    accelerationVariances = np.array([0.025, 0.025, 0.025, 0.6, 0.6, 0.6]).reshape((-1, 1))
    measurementNoiseVariances = np.array([9, 9]).reshape((-1, 1))

    
    """ 
    Build a new map
    """
    slamap =  Map(K, distCoeffs, frameSize, patchSize, minDensity, maxDensity, failTolerance, 
        accelerationVariances, measurementNoiseVariances)


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return -1

    window = "MonoSLAM"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    while True:
        
        frame = readFrame(cap)
        t0 = time.time()
        cv2.imshow(window, frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        init = slamap.initMap(gray, patternSize, squareSize, var)

        # if init:
        #     break

        if cv2.waitKey(1) == 27:
            break

    if not init:
        return 0

    while True:

        print("Visible features: ")
        print(slamap.numVisibleFeatures)
        print(len(slamap.features))
        print('\n')

        slamap.trackNewCandidates(gray)

        frame = readFrame(cap)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        slamap.predict(dt)
        slamap.update(gray, frame)

        cv2.imshow(window, frame)

        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()

