from camera import camera
from feature import Feature, FeatureNew
from quaternion import *
from utils import *

import numpy as np
import cv2

class Map:
    def __init__(self, K, distCoeffs, frameSize, patchSize_, minFeatureDensity_, maxFeatureDensity_, failTolerance_, accelerationVariances, measurementNoiseVariances):
        self.K = K
        self.distCoeffs = distCoeffs
        self.frameSize = frameSize
        self.patchSize = patchSize_
        self.minFeatureDensity = minFeatureDensity_
        self.maxFeatureDensity = maxFeatureDensity_
        self.failTolerance = failTolerance_
        self.measurementNoiseVariances = measurementNoiseVariances

        self.camera = camera(self.K, self.distCoeffs, self.frameSize)

        self.A = np.diag(accelerationVariances.flatten())
        self.R = measurementNoiseVariances

        self.x = np.zeros(13)
        self.P = np.zeros((13, 13))

        self.posIdx = np.arange(0, 3)
        self.oriIdx = np.arange(3, 7)
        self.velIdx = np.arange(7, 10)
        self.omgIdx = np.arange(10,13)

        self.features = []
        self.candidates = []
        self.inviewPos = np.empty((0, 2))
        self.numVisibleFeatures = 0


    def initMap(self, frame, patternSize, squareSize, var):
        w = patternSize[0]
        h = patternSize[1]

        criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        patternFound, imageCorners = cv2.findChessboardCorners(frame, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not patternFound:
            return False

        imageCorners = cv2.cornerSubPix(frame, imageCorners, (11, 11), (-1, -1), criteria)
        # image = cv2.drawChessboardCorners(frame, (7,9), imageCorners, True)

        worldCorners = np.empty((0, 3))
        for i in range(h):
            for j in range(w):
                worldCorners = np.vstack((worldCorners, np.array([squareSize * i, squareSize * j, 0])))

        imageCorners = np.squeeze(imageCorners)
        poseFound, rvec, tvec = cv2.solvePnP(worldCorners, imageCorners, self.K, self.distCoeffs, False, cv2.SOLVEPNP_ITERATIVE)

        if not poseFound:
            return False

        R, _ = cv2.Rodrigues(rvec)
        t = -R.T @ tvec

        q = quaternionInv(computeQuaternion(rvec))

        # Update the map state vector with the camera state vector (zero linear and angular velocities)
        self.x[self.posIdx] = t.flatten()
        self.x[self.oriIdx] = q.flatten()

        # Update the map covariance matrix with the camera covariance matrix
        self.P = np.diag(var)

        init = True
        
        idx = 0
        init = init and self.addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t)
        frame = drawSquare(frame, imageCorners[idx], 11, (0, 0, 255))

        idx = int(w - 1)
        init = init and self.addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t)
        frame = drawSquare(frame, imageCorners[idx], 11, (0, 0, 255))

        idx = int((h - 1) * w)
        init = init and self.addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t)
        frame = drawSquare(frame, imageCorners[idx], 11, (0, 0, 255))

        idx = int(w * h - 1)
        init = init and self.addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t)
        frame = drawSquare(frame, imageCorners[idx], 11, (0, 0, 255))

        idx = int(w * ((h - 1) / 2) + (w + 1) / 2 - 1)
        init = init and self.addInitialFeature(frame, imageCorners[idx], worldCorners[idx], R, t)
        frame = drawSquare(frame, imageCorners[idx], 11, (0, 0, 255))

        cv2.imshow('Feature Init', frame)

        if cv2.waitKey(1) == 27:
            exit()

        if not init:
            self.reset()
            return False

        return True


    # def broadcastData():
    #     self.camera.r = self.x[self.posIdx]
    #     self.camera.q = self.x[self.oriIdx]
    #     self.camera.v = self.x[self.velIdx]
    #     self.camera.w = self.x[self.omgIdx]

    #     self.camera.R = getRotationMatrix(self.camera.q);
    #     self.camera.P = self.P[0:13, 0:13]

    #     for i in range(self.feature.shape[0]):
    #         self.features[i].pos = x[(13 + 3*i):(13 + 3*i + 3)]
    #         self.features[i].P = P[(13 + 3*i):(13 + 3*i + 3), (13 + 3*i):(13 + 3*i + 3)]


    def trackNewCandidates(self, frame):

        # If there are enough visible features there is no need to detect new ones
        if self.numVisibleFeatures >= self.minFeatureDensity:
            return False

        # The same applies if there are few features but candidates are already being tracked
        if not len(self.candidates) == 0:
            return False

        corners = self.findCorners(frame).squeeze()

        # If no good corners were found just pass
        if corners.shape[0] == 0:
            return False

        # Undistort and normalize the pixels
        undistorted = cv2.undistortPoints(corners, self.camera.K, self.camera.distCoeffs)

        #Add the z component and reshape to 3xN
        undistorted = undistorted.squeeze().T
        undistorted = np.vstack((undistorted, np.ones(undistorted.shape[1])))

        # Get camera translation and rotation
        t = self.x[self.posIdx]
        R = getRotationMatrix(self.x[self.oriIdx])

        # Compute the directions (in world coordinates) of the lines that contain the features
        directions = R @ undistorted

        depthInterval = np.array([0.05, 5])
        depthSamples = 100

        # Normalize the direction vectors and add new pre-initialized features
        for i in range(directions.shape[1]):
            cv2.normalize(directions[:, i], directions[:, i])
            pixel = corners[i]
            self.candidates.append(Feature(frame, buildSquare(pixel, self.patchSize), R.T, t, directions[:, i], depthInterval, depthSamples))
       
        return True


    def predict(self, dt):
        F = self.computeProcessMatrix(dt)
        N = self.computeProcessNoiseMatrix(dt, F)

        self.P = F @ self.P @ F.T + N

        self.applyMotionModel(dt)


    def update(self, gray, frame):

        residuals = np.empty((0, 2))
        matchedInviewIndices = []
        failedInviewIndices = []
        failedIndices = []

        H, inviewIndices = self.computeMeasurementMatrix()

        # If there are no predicted in sight features skip the update step
        if len(inviewIndices)==0:

            self.numVisibleFeatures = 0
            return

        N = np.zeros((H.shape[0], H.shape[0]))

        for i in range(int(N.shape[0]/2)):
            N[2*i, 2*i] = self.R[0, 0]
            N[2*i + 1, 2*i + 1] = self.R[1, 0]

        # Compute the innovation covariance matrix
        S = H @ self.P @ H.T + N

        # Compute the error ellipses of the predicted in sight features pixel positions
        ellipses = computeEllipses(self.inviewPos, S)

        for i in range(ellipses.shape[0]):
            idx = inviewIndices[i]
            self.features[idx].matchingAttempts += 1

            p = self.x[13 + 3*idx: 13 + 3*idx + 3]
            n = self.features[idx].normal
            view1 = self.features[idx].image
            patch1 = self.features[idx].roi
            R1 = self.features[idx].R
            t1 = self.features[idx].t
            R2 = getRotationMatrix(self.x[3:7]).T
            t2 = self.x[0:3]

            # Compute its appearance from the current camera pose
            templ, u, v = self.camera.warpPatch(p, n, view1, patch1, R1, t1, R2, t2)
            colorTempl = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            frame = drawTemplate(frame, colorTempl, self.inviewPos[i], u, v)

            # If the template size is zero the feature is far away and not visible
            if templ.shape.all()==0:
                self.features[idx].matchingFails += 1
                failedInviewIndices.append(i)
                failedIndices.append(idx)

            else:
                # Compute the rectangular region to match this template
                roi = getBoundingBox(ellipses[i], gray.shape)

                x0 = max(roi[1] - u, 0)
                y0 = max(roi[0] - v, 0)
                x1 = min(roi[1] + roi[3] + templ.shape[1] - u, gray.shape[1] - 1)
                y1 = min(roi[0] + roi[2] + templ.shape[0] - v, gray.shape[0] - 1)

                roi[0] = y0
                roi[1] = x0
                roi[2] = y1 - y0 + 1
                roi[3] = x1 - x0 + 1

                image = gray[roi[0]:roi[2], roi[1]:roi[3]]

                # Match the template
                ccorr = cv2.matchTemplate(image, templ, cv2.TM_CCORR_NORMED)

                # Get the observation value
                _, maxVal, _, maxLoc = cv2.minMaxLoc(ccorr)

                if maxVal > 0.5:
                    px = maxLoc[1] + roi[1] + u
                    py = maxLoc[0] + roi[0] + v

                    residuals = np.vstack((residuals, np.array([px - inviewPos[i][0], py - inviewPos[i][1]])))

                    matchedInviewIndices.append(i)

                else:

                    self.features[idx].matchingFails += 1
                    failedInviewIndices.append(i)
                    failedIndices.append(idx)

        self.numVisibleFeatures = len(matchedInviewIndices)

        if self.numVisibleFeatures == 0:
            drawFeatures(frame, matchedInviewIndices, failedInviewIndices, ellipses)
            removeBadFeatures(failedInviewIndices, failedIndices)
            return

        # Compute measurement residual
        y = residuals.T
        y = y.reshape(1).T

        if not len(failedInviewIndices) == 0:

            # Reshape measurement matrix H and innovation covariance S
            H = removeRows(H, failedInviewIndices, 2);
            S = removeRowsCols(S, failedInviewIndices, 2);


        # Compute Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the map state and the map state covariance matrix
        self.x += K @ Y
        self.P -= K @ H @ self.P

        frame = drawFeatures(frame, matchedInviewIndices, failedInviewIndices, ellipses)
        removeBadFeatures(failedInviewIndices, failedIndices)
        renormalizeQuaternion()



    def addInitialFeature(self, frame, pos2D, pos3D, R, t):
        # Copy the new feature position into the state vector
        self.x = np.append(self.x, pos3D)

        # Resize the state covariance matrix
        tmp = np.zeros((self.P.shape[0] + 3, self.P.shape[1] + 3))
        tmp[:self.P.shape[0], :self.P.shape[1]] = self.P
        self.P = tmp

        # Compute the feature patch normal
        normal = np.array([0, 0, 1])

        pixelPosition = pos2D.astype(int)
        roi = buildSquare(pixelPosition, self.patchSize)

        overflow = roi[0] < 0 or roi[1] < 0 or roi[2] > frame.shape[0] or roi[3] > frame.shape[1]

        if overflow:
            return False

        self.features.append(FeatureNew(frame, roi, normal, R, t))
        self.inviewPos = np.vstack((self.inviewPos, pos2D))

        self.numVisibleFeatures += 1

        return True


    def reset(self):
        self.features = []
        self.candidates = []
        self.inviewPos = np.empty((0, 2))

        self.numVisibleFeatures = 0

        self.x = np.zeros(13)
        self.P = np.zeros((13, 13))


    def findCorners(self, frame):

        maxCorners = self.maxFeatureDensity - self.numVisibleFeatures
        minDistance = 60
        qualityLevel = 0.2

        mask = np.zeros_like(frame)

        # Set a margin around the mask to enclose the detected corners inside
        # a rectangle and avoid premature corner loss due to camera movement.
        # Pad should be larger than patchSize/2 so that new features patches
        # fit inside the camera frame.
        pad = 30

        mask[pad:(mask.shape[0] - pad), pad:(mask.shape[1] - pad)] = 255

        for i in range(self.inviewPos.shape[0]):

            roi = buildSquare(self.inviewPos[i], 2 * minDistance)

            roi[0] = max(0, roi[0])
            roi[1] = max(0, roi[1])
            roi[2] = min(frame.shape[0], roi[2])
            roi[3] = min(frame.shape[1], roi[3])

            # Update the mask to reject the region around the ith inview feature
            mask[roi[0]:roi[2], roi[1]:roi[3]] = 0

        corners = cv2.goodFeaturesToTrack(frame, maxCorners, qualityLevel, minDistance, mask)

        return corners


    def applyMotionModel(self, dt):
        self.x[0:3] += self.x[7:10] * dt

        if (not self.x[10] == 0) and (not self.x[11] == 0) and (not self.x[12] == 0):
            self.x[3:7] = quaternionMultiply(self.x[3:7], computeQuaternion(self.x[10:13] * dt))


    def computeProcessMatrix(self, dt):

        F = np.eye(self.P.shape[0])

        F[0, 7] = dt
        F[1, 8] = dt
        F[2, 9] = dt

        Q1 = np.zeros((4, 4))

        #Current pose quaternion (q_old)
        a1 = self.x[3]
        b1 = self.x[4]
        c1 = self.x[5]
        d1 = self.x[6]

        # Row 1
        Q1[0, 0] =  a1
        Q1[0, 1] = -b1
        Q1[0, 2] = -c1
        Q1[0, 3] = -d1

        # Row 2
        Q1[1, 0] =  b1
        Q1[1, 1] =  a1
        Q1[1, 2] = -d1
        Q1[1, 3] =  c1

        # Row 3
        Q1[2, 0] =  c1
        Q1[2, 1] =  d1
        Q1[2, 2] =  a1
        Q1[2, 3] = -b1

       # // Row 4
        Q1[3, 0] =  d1
        Q1[3, 1] = -c1
        Q1[3, 2] =  b1
        Q1[3, 3] =  a1

        Q2 = np.zeros((4,3))

        w1 = self.x[10]
        w2 = self.x[11]
        w3 = self.x[12]

        if w1 == 0 and w2 == 0 and w3 == 0:
            # Compute D(q(w*dt))/D(w_old)
        
            Q2[1, 0] = 0.5 * dt
            Q2[2, 1] = 0.5 * dt
            Q2[3, 2] = 0.5 * dt

        else: 
            # Set D(q_new)/D(q_old)

            # Rotation quaternion q(w*dt)
            q = computeQuaternion(self.x[10:13] * dt)

            a2 = q[0]
            b2 = q[1]
            c2 = q[2]
            d2 = q[3]

            # Row 1
            F[3, 3] =  a2
            F[3, 4] = -b2
            F[3, 5] = -c2
            F[3, 6] = -d2

            # Row 2
            F[4, 3] =  b2
            F[4, 4] =  a2
            F[4, 5] =  d2
            F[4, 6] = -c2

            # Row 3
            F[5, 3] =  c2
            F[5, 4] = -d2
            F[5, 5] =  a2
            F[5, 6] =  b2

            # Row 4
            F[6, 3] =  d2
            F[6, 4] =  c2
            F[6, 5] = -b2
            F[6, 6] =  a2

            # Compute D(q(w*dt))/D(w_old)

            w11 = w1 * w1
            w22 = w2 * w2
            w33 = w3 * w3

            norm2 = w11 + w22 + w33
            norm = np.sqrt(norm2)

            c = np.cos(0.5 * dt * norm)
            s = np.sin(0.5 * dt * norm)

            z = s / norm

            dq1dw_ = - 0.5 * dt * z
            dq_dw_ = 0.5 * dt * c / norm2 - z / norm2

            w12 = w1 * w2;
            w13 = w1 * w3;
            w23 = w2 * w3;

            # Row 1
            Q2[0, 0] = dq1dw_ * w1
            Q2[0, 1] = dq1dw_ * w2
            Q2[0, 2] = dq1dw_ * w3

            # Row 2
            Q2[1, 0] = dq_dw_ * w11 + z
            Q2[1, 1] = dq_dw_ * w12
            Q2[1, 2] = dq_dw_ * w13

            # Row 3
            Q2[2, 0] = dq_dw_ * w12
            Q2[2, 1] = dq_dw_ * w22 + z
            Q2[2, 2] = dq_dw_ * w23

            # Row 4
            Q2[3, 0] = dq_dw_ * w13
            Q2[3, 1] = dq_dw_ * w23
            Q2[3, 2] = dq_dw_ * w33 + z

        # Set D(q_new)/D(w_old)
        F[3:7, 10:13] = Q1 @ Q2

        return F


    def computeProcessNoiseMatrix(self, dt, F):

        W0 = np.zeros((7, 6))

        # Set D(r_new)/D(V)
        W0[0, 0] = dt
        W0[1, 1] = dt
        W0[2, 2] = dt

        # Set D(q_new)/D(W)
        W0[3:7, 3:6] = F[3:7, 10:13]

        # Compute W * Q * W^t
        Q = self.A * dt * dt
        W0_Q = W0 @ Q
        W0_Q_t = W0_Q.T
        W0_Q_W0 = W0_Q @ W0.T

        N = np.zeros_like(self.P)

        N[0:7, 0:7] = W0_Q_W0
        N[0:7, 7:13] = W0_Q
        N[7:13, 0:7] = W0_Q_t
        N[7:13, 7:13] = Q

        return N


    def computeMeasurementMatrix(self):
        inviewIndices = []
        self.inviewPos = np.empty((0, 2))

        # Get the predicted camera rotation (from world basis to camera basis)
        R = getRotationMatrix(self.x[3:7]).T

        # Get the predicted camera position (in world coordinates)
        t = self.x[0:3]

        points3D_ = np.zeros((len(self.features), 3))

        # Set a vector of features positions
        for i in range(points3D_.shape[0]):
            points3D_[i] = self.x[13 + 3*i:13 + 3*i + 3]

        # Project all the features positions to the current view
        points2D_ = self.camera.projectPoints(R, t, points3D_).squeeze()

        points3D = np.empty((0, 3))

        for i in range(points2D_.shape[0]):

            if points2D_[i][0]>=0 and points2D_[i][0]< self.camera.frameSize[0] and points2D_[i][1]>=0 and points2D_[i][1]< self.camera.frameSize[1]:
                inviewIndices.append(i)
                self.inviewPos = np.vstack((self.inviewPos, points2D_[i]))
                points3D = np.vstack((points3D, points3D_[i]))

        H = np.zeros((2 * len(inviewIndices), self.x.shape[0]))

        R1, R2, R3, R4 = getRotationMatrixDerivatives(self.x[3:7])

        fx = self.camera.K[0, 0]
        fy = self.camera.K[1, 1]

        k1 = self.camera.distCoeffs[0, 0]
        k2 = self.camera.distCoeffs[0, 1]
        p1 = self.camera.distCoeffs[0, 2]
        p2 = self.camera.distCoeffs[0, 3]
        k3 = self.camera.distCoeffs[0, 4]

        for i in range(int(H.shape[0]/2)):

            pt = points3D[i] - t

            pCam = R @ pt

            xCam = pCam[0]
            yCam = pCam[1]
            zCam = pCam[2]

            zCam2 = zCam * zCam

            pBar1 = R1 @ pt
            pBar2 = R2 @ pt
            pBar3 = R3 @ pt
            pBar4 = R4 @ pt

            # D(xn)/D(r): derivatives of normalized camera coordinate xn with respect to camera position
            dxndr1 = (R[2, 0] * xCam - R[0, 0] * zCam) / zCam2
            dxndr2 = (R[2, 1] * xCam - R[0, 1] * zCam) / zCam2
            dxndr3 = (R[2, 2] * xCam - R[0, 2] * zCam) / zCam2

            # D(xn)/D(q): derivatives of normalized camera coordinate xn with respect to camera orientation
            dxndq1 = (pBar1[0] * zCam - pBar1[2] * xCam) / zCam2
            dxndq2 = (pBar2[0] * zCam - pBar2[2] * xCam) / zCam2
            dxndq3 = (pBar3[0] * zCam - pBar3[2] * xCam) / zCam2
            dxndq4 = (pBar4[0] * zCam - pBar4[2] * xCam) / zCam2

            # D(yn)/D(r): derivatives of normalized camera coordinate yn with respect to camera position
            dyndr1 = (R[2, 0] * yCam - R[1, 0] * zCam) / zCam2
            dyndr2 = (R[2, 1] * yCam - R[1, 1] * zCam) / zCam2
            dyndr3 = (R[2, 2] * yCam - R[1, 2] * zCam) / zCam2

            # D(yn)/D(q): derivatives of normalized camera coordinate yn with respect to camera orientation
            dyndq1 = (pBar1[1] * zCam - pBar1[2] * yCam) / zCam2
            dyndq2 = (pBar2[1] * zCam - pBar2[2] * yCam) / zCam2
            dyndq3 = (pBar3[1] * zCam - pBar3[2] * yCam) / zCam2
            dyndq4 = (pBar4[1] * zCam - pBar4[2] * yCam) / zCam2

            xn = xCam / zCam
            yn = yCam / zCam

            # D(r2)/D(r): derivatives of r^2 = xn^2 + yn^2 with respect to camera position
            dr2dr1 = 2 * (xn * dxndr1 + yn * dyndr1)
            dr2dr2 = 2 * (xn * dxndr2 + yn * dyndr2)
            dr2dr3 = 2 * (xn * dxndr3 + yn * dyndr3)

            # D(r2)/D(q): derivatives of r^2 = xn^2 + yn^2 with respect to camera orientation
            dr2dq1 = 2 * (xn * dxndq1 + yn * dyndq1)
            dr2dq2 = 2 * (xn * dxndq2 + yn * dyndq2)
            dr2dq3 = 2 * (xn * dxndq3 + yn * dyndq3)
            dr2dq4 = 2 * (xn * dxndq4 + yn * dyndq4)

            r2 = xn * xn + yn * yn
            r4 = r2 * r2
            r6 = r4 * r2

            a = 1 + k1*r2 + k2*r4 + k3*r6
            b = k1 + 2*k2*r2 + 3*k3*r4

            ddr1 = 2 * (xn * dyndr1 + yn * dxndr1)
            ddr2 = 2 * (xn * dyndr2 + yn * dxndr2)
            ddr3 = 2 * (xn * dyndr3 + yn * dxndr3)

            ddq1 = 2 * (xn * dyndq1 + yn * dxndq1)
            ddq2 = 2 * (xn * dyndq2 + yn * dxndq2)
            ddq3 = 2 * (xn * dyndq3 + yn * dxndq3)
            ddq4 = 2 * (xn * dyndq4 + yn * dxndq4)

            # D(u)/D(r)
            dudr1 = fx * (dxndr1*a + xn*dr2dr1*b + p1*ddr1 + p2*(dr2dr1 + 4*xn*dxndr1))
            dudr2 = fx * (dxndr2*a + xn*dr2dr2*b + p1*ddr2 + p2*(dr2dr2 + 4*xn*dxndr2))
            dudr3 = fx * (dxndr3*a + xn*dr2dr3*b + p1*ddr3 + p2*(dr2dr3 + 4*xn*dxndr3))

            # D(u)/D(x)
            dudx1 = - dudr1
            dudx2 = - dudr2
            dudx3 = - dudr3

            # D(u)/D(q)
            dudq1 = fx * (dxndq1*a + xn*dr2dq1*b + p1*ddq1 + p2*(dr2dq1 + 4*xn*dxndq1))
            dudq2 = fx * (dxndq2*a + xn*dr2dq2*b + p1*ddq2 + p2*(dr2dq2 + 4*xn*dxndq2))
            dudq3 = fx * (dxndq3*a + xn*dr2dq3*b + p1*ddq3 + p2*(dr2dq3 + 4*xn*dxndq3))
            dudq4 = fx * (dxndq4*a + xn*dr2dq4*b + p1*ddq4 + p2*(dr2dq4 + 4*xn*dxndq4))

            # D(v)/D(r)
            dvdr1 = fy * (dyndr1*a + yn*dr2dr1*b + p2*ddr1 + p1*(dr2dr1 + 4*yn*dyndr1))
            dvdr2 = fy * (dyndr2*a + yn*dr2dr2*b + p2*ddr2 + p1*(dr2dr2 + 4*yn*dyndr2))
            dvdr3 = fy * (dyndr3*a + yn*dr2dr3*b + p2*ddr3 + p1*(dr2dr3 + 4*yn*dyndr3))

            # D(v)/D(x)
            dvdx1 = - dvdr1
            dvdx2 = - dvdr2
            dvdx3 = - dvdr3

            # D(v)/D(q)
            dvdq1 = fy * (dyndq1*a + yn*dr2dq1*b + p2*ddq1 + p1*(dr2dq1 + 4*yn*dyndq1))
            dvdq2 = fy * (dyndq2*a + yn*dr2dq2*b + p2*ddq2 + p1*(dr2dq2 + 4*yn*dyndq2))
            dvdq3 = fy * (dyndq3*a + yn*dr2dq3*b + p2*ddq3 + p1*(dr2dq3 + 4*yn*dyndq3))
            dvdq4 = fy * (dyndq4*a + yn*dr2dq4*b + p2*ddq4 + p1*(dr2dq4 + 4*yn*dyndq4))

            H[2*i, 0] = dudr1
            H[2*i, 1] = dudr2
            H[2*i, 2] = dudr3
            H[2*i, 3] = dudq1
            H[2*i, 4] = dudq2
            H[2*i, 5] = dudq3
            H[2*i, 6] = dudq4
            H[2*i, 13 + 3*i] = dudx1
            H[2*i, 13 + 3*i+1] = dudx2
            H[2*i, 13 + 3*i+2] = dudx3
            H[2*i+1, 0] = dvdr1
            H[2*i+1, 1] = dvdr2
            H[2*i+1, 2] = dvdr3
            H[2*i+1, 3] = dvdq1
            H[2*i+1, 4] = dvdq2
            H[2*i+1, 5] = dvdq3
            H[2*i+1, 6] = dvdq4
            H[2*i+1, 13 + 3*i] = dvdx1
            H[2*i+1, 13 + 3*i+1] = dvdx2
            H[2*i+1, 13 + 3*i+2] = dvdx3

        return H, inviewIndices


    def drawFeatures(self, frame, matchedInviewIndices, failedInviewIndices, ellipses, drawEllipses=True):
        if drawEllipses:
            # Draw matched features ellipses (red)
            for i in range(len(matchedInviewIndices)):

                idx = matchedInviewIndices[i]
                frame = drawEllipse(frame, ellipses[idx], (0, 0, 255))
            

            # Draw failed features ellipses (blue)
            for i in range(len(failedInviewIndices)):

                idx = failedInviewIndices[i]
                frame = drawEllipse(frame, ellipses[idx], (255, 0, 0))

        else:

            # Draw matched features (red)
            for i in range(len(matchedInviewIndices)):

                idx = matchedInviewIndices[i]
                frame = drawSquare(frame, self.inviewPos[idx], 11, (0, 0, 255))

    def removeBadFeatures(self, failedInviewIndices, failedIndices):

        badFeaturesVecIndices = []  # Indices to remove from features, x and P
        badFeaturesPosIndices = []  # Indices to remove from inviewPos

        for i in range(len(failedIndices)):

            idx = failedIndices[i]
            j = failedInviewIndices[i]

            numFails = self.features[idx].matchingFails
            numAttemps = self.features[idx].matchingAttempts

            if (numFails/numAttemps) > failTolerance:
                badFeaturesVecIndices.append(idx)
                badFeaturesPosIndices.append(j)

        self.features = removeIndicesLst(self.features, badFeaturesVecIndices)
        self.inviewPos = removeIndicesArr(self.inviewPos, badFeaturesPosIndices)

        x_Features = self.x[13:]
        P_Features = self.P[13:, 13:]

        x_Features = removeRows(x_Features, badFeaturesVecIndices, 3)
        P_Features = removeRowsCols(P_Features, badFeaturesVecIndices, 3)


    def renormalizeQuaternion(self):

        q = self.x[3:7]

        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        q11 = q1 * q1
        q22 = q2 * q2
        q33 = q3 * q3
        q44 = q4 * q4

        j1 = q22 + q33 + q44
        j2 = q11 + q33 + q44
        j3 = q11 + q22 + q44
        j4 = q11 + q22 + q33

        j12 = -q1 * q2
        j13 = -q1 * q3
        j14 = -q1 * q4
        j23 = -q2 * q3
        j24 = -q2 * q4
        j34 = -q3 * q4

        J = np.eye(self.P.shape)

        J[3, 3] = j1
        J[3, 4] = j12
        J[3, 5] = j13
        J[3, 6] = j14

        J[4, 3] = j12
        J[4, 4] = j2
        J[4, 5] = j23
        J[4, 6] = j24

        J[5, 3] = j13
        J[5, 4] = j23
        J[5, 5] = j3
        J[5, 6] = j34

        J[6, 3] = j14
        J[6, 4] = j24
        J[6, 5] = j34
        J[6, 6] = j4

        qnorm = norm(q)

        J[3:7, 3:7] /= qnorm * qnorm * qnorm

        # Normalize quaternion
        normalize(q, q);

        # Update covariance matrix
        P = J @ P @ J.T
