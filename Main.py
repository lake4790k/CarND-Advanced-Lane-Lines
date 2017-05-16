import numpy as np
import cv2
import glob
import logging
import pickle
import os.path
import matplotlib.pyplot as plt

logging.debug('Advanced lane finding initializing')


class Processor:

    def process(self):
        pass

    def show(self, img):
        img_ = self.process(img)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('original')
        ax1.imshow(img)
        ax2.set_title('processed')
        if img_.shape[2]==1:
            ax2.imshow(img_, cmap='gray')
        else:
            ax2.imshow(img_)


class Calibration(Processor):

    def __init__(self):
        self.mtx = []
        self.dist = []

        if os.path.isfile('./camera_cal/calib.mtx'):
            with open('./camera_cal/calib.mtx', 'rb') as f:
                self.mtx = pickle.load(f)
            with open('./camera_cal/calib.dist', 'rb') as f:
                self.dist = pickle.load(f)
            logging.info('loaded calibration data')
        else:
            self.calibrate()

    def calibrate(self):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        images = glob.glob('./camera_cal/calibration*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            logging.debug('calibrating %s', fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        img_size = img.shape[0:2]
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        logging.info('Computed calibration matrix')
        with open('./camera_cal/calib.mtx', 'wb') as f:
            pickle.dump(self.mtx, f)
        with open('./camera_cal/calib.dist', 'wb') as f:
            pickle.dump(self.dist, f)

    def process(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class BinaryThreshold(Processor):

    def __init__(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:, :, 1]
        s_channel = hsv[:, :, 2]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary


class PerpectiveTransform(Processor):

    def __init__(self):
        pass

    def process(self, img):
        pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    calibration = Calibration()
