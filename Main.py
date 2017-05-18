import numpy as np
import cv2
import glob
import logging
import pickle
import os.path
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
logging.debug('Advanced lane finding initializing')


class Data:
    def __init__(self):
        self.rawImg = None
        self.undistImg = None
        self.binaryImg = None
        self.warpedImg = None

        self.M = None
        self.Minv = None

        self.ploty = np.array([])
        self.left_fitx = np.array([])
        self.right_fitx = np.array([])
        self.left_fit = np.array([])
        self.right_fit = np.array([])

        self.left_curverad = None
        self.right_curverad = None


class Processor:
    def __init__(self, data: Data):
        self.data = data

    def process(self, img: np.array) -> np.array:
        return img

    def show(self, img: np.array) -> np.array:
        img_ = self.process(img)
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.show2(img, img_)
        return img_

    def show2(self, img: np.array, img_: np.array):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('original')
        if len(img.shape) < 3:
            ax1.imshow(img, cmap='gray')
        else:
            ax1.imshow(img)
        ax2.set_title('processed')
        if len(img_.shape) < 3:
            ax2.imshow(img_, cmap='gray')
        else:
            ax2.imshow(img_)


class Calibration(Processor):
    def __init__(self, data: Data):
        Processor.__init__(self, data)
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

        img_size = 0
        for fname in images:
            img = cv2.imread(fname)
            img_size = img.shape[0:2]
            logging.debug('calibrating %s', fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        logging.info('Computed calibration matrix')
        with open('./camera_cal/calib.mtx', 'wb') as f:
            pickle.dump(self.mtx, f)
        with open('./camera_cal/calib.dist', 'wb') as f:
            pickle.dump(self.dist, f)

    def process(self, img: np.array) -> np.array:
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class BinaryThreshold(Processor):
    def __init__(self, data: Data, s_thresh=(170, 255), sx_thresh=(20, 100)):
        Processor.__init__(self, data)
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def process(self, img: np.array) -> np.array:
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


class PerspectiveTransform(Processor):
    def __init__(self, data: Data):
        Processor.__init__(self, data)
        self.src = np.float32([[550, 450], [750, 450], [1200, 700], [150, 700]])
        self.dst = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
        self.data.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.data.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def process(self, img: np.array) -> np.array:
        return cv2.warpPerspective(img, self.data.M, (img.shape[1], img.shape[0]))

    def show(self, img: np.array) -> np.array:
        img_ = self.process(img)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            cv2.polylines(img, np.int32([self.src]), True, [255, 255, 255], 5)

        self.show2(img, img_)
        return img_


class FindLanes(Processor):
    def __init__(self, data: Data, nwindows=9):
        Processor.__init__(self, data)
        self.nwindows = nwindows
        self.nonzeroy = None
        self.nonzerox = None
        self.out_img = None
        self.left_lane_inds = None
        self.right_lane_inds = None

    def process(self, img: np.array) -> np.array:
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        self.out_img = np.dstack((img, img, img)) * 255

        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(img.shape[0] / self.nwindows)

        nonzero = img.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 100
        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)

        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.data.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        self.data.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.data.ploty = ploty
        self.data.left_fit = left_fit
        self.data.right_fit = right_fit
        return img

    def show(self, img: np.array):
        self.process(img)

        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        plt.figure(figsize=(20, 6))
        plt.imshow(self.out_img)
        plt.plot(self.data.left_fitx, self.data.ploty, color='yellow')
        plt.plot(self.data.right_fitx, self.data.ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)


class NextFindLanes(Processor):
    def __init__(self, data: Data):
        Processor.__init__(self, data)
        self.margin = 100
        self.nonzeroy = None
        self.nonzerox = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.leftx = None
        self.rightx = None
        self.lefty = None

    def process(self, img: np.array) -> np.array:
        nonzero = img.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        left_fit = self.data.left_fit
        right_fit = self.data.right_fit

        self.left_lane_inds = \
            ((self.nonzerox > (left_fit[0] * (self.nonzeroy ** 2) +
                               left_fit[1] * self.nonzeroy +
                               left_fit[2] - self.margin)) &
             (self.nonzerox < (left_fit[0] * (self.nonzeroy ** 2) +
                               left_fit[1] * self.nonzeroy +
                               left_fit[2] + self.margin)))

        self.right_lane_inds = \
            ((self.nonzerox > (right_fit[0] * (self.nonzeroy ** 2) +
                               right_fit[1] * self.nonzeroy +
                               right_fit[2] - self.margin)) &
             (self.nonzerox < (right_fit[0] * (self.nonzeroy ** 2) +
                               right_fit[1] * self.nonzeroy +
                               right_fit[2] + self.margin)))

        self.leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        self.data.left_fit = np.polyfit(lefty, self.leftx, 2)
        self.data.right_fit = np.polyfit(righty, self.rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.data.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        self.data.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.data.ploty = ploty
        return img

    def show(self, img: np.array):
        self.process(img)

        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)

        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        left_fitx = self.data.left_fitx
        right_fitx = self.data.right_fitx
        ploty = self.data.ploty
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure(figsize=(20, 6))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)


class Curvature(Processor):
    def __init__(self, data: Data):
        Processor.__init__(self, data)
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

    def process(self, img: np.array) -> np.array:
        ploty = self.data.ploty
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.data.left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.data.right_fitx * self.xm_per_pix, 2)

        self.data.left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                                    np.absolute(2 * left_fit_cr[0])
        self.data.right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                                    np.absolute(2 * right_fit_cr[0])
        return img

    def show(self, img: np.array):
        self.process(img)
        print(self.data.left_curverad, 'm', self.data.right_curverad, 'm')


if __name__ == '__main__':
    logging.info("STARTING")
