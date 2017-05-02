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

[histogram]: ./output_images/histogram.png
[blind_search]: ./output_images/blind_search.png
[next_search]: ./output_images/next_search.png


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

Finally, I combined both channel with the x gradients of the image using the Sobel operator. The gradients are computed in the `helpers.gradx()` method.

```python
def gradx(img, kernel=11, min_thresh=50, max_thresh=100):
    # calculates derivatives in x direction:  
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    
    # creates a binary images based on the provided min and max thresholds
    sbinary[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1
    return sbinary
```
The following image sequence shows the gradient image, the combined color channels and the final binary image.

![][combined_binary]


### 4. Lane detection
I implemented two functions `helpers.blind_search()` and `helpers.next_search()` which are used for detecting lanes in each video frame. The `helpers.blind_search()` function is applied to the first frame in the video whereas `helpers.next_search()` is used for subsequent frames.

A __blind search__ is applied to the first video frame when any prior knowledge is available or when `next_search()` was not able to find any viable lanes. It works roughly as follows:
1. Compute the histogram of the lower half of the frame. Find the two highest peaks which are a good indicator for the x positions of the lane lines. The x positions are the starting point for our lane line search. The image below shows the histogram with the two peaks. 

![][histogram]

2. Slide a window around the left and right x position over the image in y direction from the bottom to the top. This is shown in the next image. The window is recentered if more the n=50 pixels were found within the window. The found pixels are stored in list. Eventually, a second order polynomial fit is calculated.

![][blind_search]

Now we now where the lines are. We use this information when we search for the lines in the subsequent frames. We use the line positions from the previous frame and search in a surrounding area for the next lines. Next image shows this approach:

![][next_search]

I have implemented two classes __LineSegment__ and __Line__. You find both classes in the [lane.py][lane.py] file. A __LineSegment__ instance stores the information about one detected line in one frame as well as the curvature radius and the vehicle position. A __Line__ exists for the left and right lane line. It keeps the last n detected __LineSegments__. The __Lane__ class provides a sanity check method `is_valid_line()` which checks whether a __LineSegment__ is valid. It also contains the `get_smoothed_line()` method. This method computes a fit over detected lane line pixels of the last n frames. A __Line__ instance is reset if the `next_search()` function was not able to find viable lanes for several consecutive frames. A blind search is carried out after a __Line__ reset.

#### 5. Curvature and vehicle position calculation

The left and right curvature radius of the lane is calculated in the __LineSegment__ class. This is code.
```python
    def __calc_curvature__(self, xfitted):
        """
        Calculates the curvature of a line
        """
        ploty = np.linspace(0, 720-1, 720 )
        y_eval = np.max(ploty)
    
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, xfitted * xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])
        
        return round(curverad,1)
```

Vehicle position caluclation is implemented in `calc_vehicle_pos()` in the [helpers.py][helpers.py] file and looks as follows
```python
def calc_vehicle_pos(leftx, rightx, midpoint):
    meters_per_pixel = 3.7 / 700.0
    lane_width = round((rightx - leftx) * meters_per_pixel, 1)
        
    # lane midpoint
    lane_center = leftx + int((rightx - leftx) / 2)
    # calculate difference between lane midpoint and image midpoint which
    # is the deviation of the car to the lane midpoint
    diff = abs(lane_center - midpoint)    
    deviation = round(diff * meters_per_pixel, 2)
    
    return deviation, lane_width
```

### 6. Visualzing the detected lanes

I implemented this step in the ```project_lines()``` functions which can also be found in the [helpers.py][helpers.py] file. What this function does can be explained as follows. It takes the undistorted frame, the warped image (birds eye view), the line pixels for the left and right lane and the inverse transformation matrix as input. Based on the warped image an empty image is created. With the line pixels a polygone is drawn on it. Then this image is transformed back into the original perspective. Finally, a weighted image from the undistorted and unwarped image is created.

```python
def project_lines(undist, warped, left_fitx, right_fitx, Minv):
    ploty = np.linspace(0, 720-1, 720 )
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result
```

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](./project_video_result.mp4)
This is the [link to my challenge result](./challenge_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
