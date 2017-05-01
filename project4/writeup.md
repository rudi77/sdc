# Advanced Lane Finding Project #

This is the fourth project of the sdc course. In this project lane boundaries shall be detected in a video provided by udacity. Additionaly lane curvature and vehicale position values shall be estimated.

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted_chessboard]: ./output_images/distorted_chessboard.png "Distorted"
[distorted_chessboard_with_corners]: ./output_images/distorted_chessboard_with_corners.png "Distorted"
[undistorted_chessboard]: ./output_images/undistorted_chessboard.png "Undistorted"

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[calibration_images]: ./camera_cal
[calibration_tutorial]: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

[helpers.py]: ./helpers.py
[lane.py]: ./lane.py
[pipeline.py]: ./pipeline
[exploration.ipynb]: ./exploration.ipynb
## Files in this repository
This repository contains the following files.
- [pipeline.py][pipeline.py] contains the image processing steps (the pipeline) for finding lanes on the road. This pipeline is applied to each video frame
- [helpers.py][helpers.py] contains all functions that are used in each image processing.
- [lane.py][lane.py] contains the Lane and LaneSegment classes which describe the lanes that are found in video frames.
- [exploration.ipynb][exploration.ipynb] contains my ipython notebook which I used for data exploration.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration
Distorted images produced by pinhole cameras can be corrected by applying camera calibration techniques. In this project I use the [black-white chessboard][calibration_tutorial] approach. The camera calibration code is implemented in the `helpers.camera_calibration()` functions in the [helpers.py][helpers.py] file
In principle this approach works as follows:
1. Images from a chessboard must be taken from a static camera. Images must be take from different perspectives. Therefore the chessboard is placed at differnt locations and orientations. I used these [images][calibration_images] which were also provided by Udacity.
2. Then we need the coordinates (x,y,z) of the chessboard corners in real world and on the image (x,y). The real world coordinates are called objectpoints and the corresponding coordinates on the image are called imagepoints.
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

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

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
