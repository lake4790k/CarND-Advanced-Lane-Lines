## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib.png "Undistorted"
[image2]: ./output_images/undist.png "Road Transformed"
[image3]: ./output_images/thres.png "Binary Example"
[image4]: ./output_images/pers.png "Warp Example"
[image4b]: ./output_images/pers2.png "Warp Example"
[image5]: ./output_images/lane.png "Fit Visual"
[image5b]: ./output_images/lane2.png "Fit Visual"
[image6]: ./output_images/final.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The logic is implemented in the `Calibration` class in `Main.py`.

I start by preparing 6x9 "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

I calculated the coefficients once and reload them from pickle file on every run.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The `Calibration` class in `Main.py` loads and applies the precomputed coefficients on the input images by calling `cv2.undistort()` in the `process` method, eg: 

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The `BinaryThreshold.process()` method in `Main.py` computes the binary image. I converted to HLS color space and used thresholded sobel transform on the L channel and thresholding on the S channel and combine the results of the two, eg: 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The `PerspectiveTransform.process()` method in `Main.py` computes the perspective warped image. I manually selected source points for the lane in the original image and precompute the trasformation matrix on instantiating this class.

I used the following pixel mapping:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 300, 0        | 
| 700, 450      | 980, 0      |
| 1100,700      | 980, 720      |
| 200,700       | 300, 720        |

The warp gives this result:

![alt text][image4]

![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I adapted the windowed method from the class in the `FindLanes.process()` method in `Main.py` and also the simpler method that searched around the previous lane position in the `NextFindLanes.process()` method in `Main.py`.

![alt text][image5]

![alt text][image5b]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The `Curvature.process()` method in `Main.py` computes the lane curvature in rads by averaging the the left and right line curvatures. The values are calculated in meter space with the suggested pixel to m conversion method adjusted to the pixel area dimensions I choose.

The `LaneOffset.process()` method in `Main.py` computes the car position in the lane by comparing the

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `DecorateWithData.process()` method in `Main.py` plots the fitted lane back to the undistorted image and also annotates the image with the computed curvature, offset and fit parameters. Seeing the numberic values on the problematic sections was essential in seeing how the computation could be fixed.

An example image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed.project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
