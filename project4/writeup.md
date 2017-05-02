# Advanced Lane Finding Project #

This is the fourth project of the sdc course. In this project lane boundaries shall be detected in a video provided by udacity. Additionally, the lane curvature radius and vehicle position shall be calculated and displayed.

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (References)

[helpers.py]: ./helpers.py
[lane.py]: ./lane.py
[pipeline.py]: ./pipeline
[exploration.ipynb]: ./exploration.ipynb
[parameters.p]: ./parameters.p

[calibration_images]: ./camera_cal
[calibration_tutorial]: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
[distorted_chessboard]: ./output_images/distorted_chessboard.png "Distorted"
[distorted_chessboard_with_corners]: ./output_images/distorted_chessboard_with_corners.png "Distorted"
[undistorted_chessboard]: ./output_images/undistorted_chessboard.png "Undistorted"
[original_image]: ./output_images/original_image.png
[undistorted_image]: ./output_images/undistorted_image.png

[default_perspective]: ./output_images/default_perspective.png
[warped]: ./output_images/warped.png
[unwarped]: ./output_images/unwarped.png

[color_to_binary]: ./output_images/color_to_binary.png
[combined_binary]: ./output_images/combined_binary.png


## Files in this repository
This repository contains the following files.
- [pipeline.py][pipeline.py] contains the image processing steps (the pipeline) for finding lanes on the road. This pipeline is applied to each video frame
- [helpers.py][helpers.py] contains the functions that are used in each image processing step.
- [lane.py][lane.py] contains the Lane and LaneSegment classes which describe the lanes that are found in a video frames.
- [exploration.ipynb][exploration.ipynb] contains my ipython notebook which I used for data exploration.
- [parameters.p][parameters.p] a pickle file which contains the calibration matrix, the distortion coefficients, and the transformation matrices M and Minv.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

## Camera Calibration
Distorted images produced by pinhole cameras can be corrected by applying camera calibration techniques. In this project I use the [black-white chessboard][calibration_tutorial] approach. The camera calibration code is implemented in the `helpers.camera_calibration()` functions in the [helpers.py][helpers.py] file
In principle this approach works as follows:
1. Images from a chessboard must be taken from a static camera. Images must be taken from different perspectives. Therefore the chessboard is placed at differnt locations and orientations. I used these [images][calibration_images] which were also provided by Udacity.
2. Then we need the real world coordinates (x,y,z) as well as the image coordinates (x,y) of the chessboard corners. The real world coordinates are called objectpoints and the corresponding coordinates on the image are called imagepoints.
    - Objectpoints can be generated using numpy's `np.mgrid()` function.
    - The chessboard corners in the image can be found with the opencv's `cv2.findChessboardCorners()` function. This function takes a grayscale image as input argument.
3. Having retrieved all objectpoints and imagepoints, we can now calculate the calibration matrix __mtx__ and distortion coefficients __dist__ which are needed to undistort an image.
4. Finally, we can now undistort an image using `cv2.undistort()`

By applying theses camera calibration steps I obtained the following results:

 Distorted                 | Distorted with corners                   | Undistorted
:-------------------------:|:----------------------------------------:|:---------------------------
![][distorted_chessboard]  |  ![][distorted_chessboard_with_corners]  | ![][undistorted_chessboard]

This is my camera calibration code:
```python
def calibrate_camera(cal_images, nx, ny):
    # arrays to store all object points and image points from all images
    objpoints = [] 
    imgpoints = []

    # prepare object points like (0,0,0), (1,0,0),(2,0,0) ... (8,4,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates
    
    img_shape = None
    
    for img_name in cal_images:
        img = mpimg.imread(img_name)
        
        if img_shape is None:
            img_shape = (img.shape[1], img.shape[0])
        
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # find chessboardcorners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    assert(len(objpoints) == len(imgpoints))
    
    # get camera calibration matrix and distoration coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    return mtx, dist
```
## Land Detection Pipeline
In the following sections I will describe my image processing pipeline that detects and visualizes road lanes. My pipeline consists
of the following steps which are applied on each single video frame.
- Image distortion correction
- Perspective transformation
- Color and gradient transformation and binarization
- Lane detection and visualization
- Curvature and vehicle position calculation

### 1. Image distortion correction
Every image is undistorted by calling the `undistort(image, mtx, dist)` function in the [helpers.py][helpers.py] file.
```python
def undistort(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)
```
This is an example of an undistorted image. First the original image is shown and then the undistorted image is presented.

 Original Image                 | Undistorted Image
:-------------------------:|:----------------------------------------
![][original_image]  |  ![][undistorted_image]  

### 2. Perspective transformation
In a second step a perspective transformation is applied to the undistorted image. In particular a bird’s-eye view transform is applied. This view is then used in a subsequent step to find and extract the road lanes.
The [helpers.py][helpers.py] file contains a `warp()` function which creates and returns a bird’s eye view representation of the image. Source and destination points are needed for a perspective transformation. The source points will be mapped on the provided destination points. The source and destination points are hardcoded into the `warp()` function.

 Source points  | Destination points     
:--------------:|:-----------------
   (710,460)    |   (1000,0)
   (1110,720)   |   (1000,720)
   (205,720)    |   (300,720)
   (575,460)   |   (300,0)

This is the corresponding pyhton function:
```python
def warp(undist):
    src = np.float32([[710,460],[1110,720],[205,720],[575,460]])
    dst = np.float32([[1000,0],[1000,720],[300,720],[300,0]])    
        
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_size = (undist.shape[1], undist.shape[0])
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M
```
There also exists an `unwarp()` function which transforms an image back to its original representation. This function is called as a final step when the detected lanes are projected on the processed video frame. The `unwarp()` function is the same as the `warp()` function but with interchanged source and destination points.

The following images show a transformation to a bird's eye view and back to the original perspective.

 Default perspective       | Bird's eye view                          | Unwarped
:-------------------------:|:----------------------------------------:|:---------------------------
![][default_perspective]   |  ![][warped]                             | ![][unwarped]

### 3.Color and gradient transformation and binarization.

I used a combination of color and gradient thresholds to generate a binary image. First, I extracted the B channel of the LAB color space and the G channel of the RGB space of the current frame in the `pipeline.process_image()` function. Then I converted them into binary images using `helpers.binary_channel` function.
```python
    # binarize image
    b_channel = cv2.cvtColor(img_birds_eye, cv2.COLOR_RGB2LAB)[:,:,2]
    b_binary = helpers.binary_channel(b_channel, min_thresh=145, max_thresh=180)    
    g_channel = img_birds_eye[:,:,1]
    g_binary = helpers.binary_channel(g_channel, min_thresh=170, max_thresh=255) 
 ```

The following image sequence shows the extracted B and G channels as well as the combination of both binary images.

![][color_to_binary]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
