# UNDER CONSTRUCTION #


# Vehicle Detection Project

This is this fifth and last project of the first term. In this project vehicles in a provided movie shall be detected. Additionaly, I've also re-used the code from project 4 to detect road lanes.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[pipeline.py]: ./pipeline.py
[classifier.py]: ./classifier.py
[dataset_generator.py]: ./dataset_generator.py
[exploration.ipynb]: ./exploration.ipynb
[writeup.md]: ./writeup.md

[helpers.py]: ./helpers.py
[lane.py]: ./lane.py
[parameters.p]: ./parameters.p

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/color_histogram.png
[image3_2]: ./output_images/color_histogram_notcar.png
[image4]: ./output_images/spatial_binning.png
[image4_2]: ./output_images/spatial_binning_notcar.png

## Files in this repository
This repository contains the following files.
#### Vehicle detection ####
- [pipeline.py][pipeline.py] contains the image processing steps (the pipeline) for finding vehicles and lanes on the road. This pipeline is applied to each video frame
- [classifier.py][classifier.py] contains the code for training a support vector machine.
- [dataset_generator.py][dataset_generator.py] contains the code for generating features like HoG, spatial bins and color histograms and dataset files
- [exploration.ipynb][exploration.ipynb] a notebook which I have used for explorative data analysis.
- [writeup.md][writeup.md] 

#### Road lanes detection ####
- [helpers.py][helpers.py] contains the functions that are used in each image processing step.
- [lane.py][lane.py] contains the Lane and LaneSegment classes which describe the lanes that are found in a video frames.
- [parameters.p][parameters.p] a pickle file which contains the calibration matrix, the distortion coefficients, and the transformation matrices M and Minv.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---

## Vehicle Detection Pipeline
In the following sections I will describe my pipeline that detects and visualizes vehicles and lane lines. The pipeline consists
of the following steps.

#### Preprocessing ####
1. Generate a dataset.
2. Use the generated dataset to train a classifier.
3. Serialize the model to a file so that it can be later used in the actual image processing pipeline

#### Image Processing Pipeline ####
1. Use a sliding window search to detect possible vehicles
2. Generate a heatmap. The heatmap is used to reduce false positives as well as to identify vehicles
3. Mark detected vehicles with a bounding box
4. Detect lane lines

## Preprocessing
Preprocessing includes dataset generation und training a classifier.

### 1. Dataset Generation
I downloaded the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) dataset which was provided by [Udacity](https://udacity.com) and copied them into two separate folders respectively. Then I iterated over each  `vehicle` and `non-vehicle` image and computed a feature vector. I implemented this step in the `create_feature_row()` function in the `dataset_generator.py` file. I also added the corresponding label/class (CAR=1, NOTCAR=0) to the end of the feature vector. Finally, I generated a [pandas](http://pandas.pydata.org/) dataframe from the feature vectors and stored the frame as csv file. This file is later used to generate training and tet sets for the classifier.

Here is an example of a randomly chosen car and non-car image.

![][image1]

#### Histogram of Oriented Gradients (HOG)
The [histogram of oriented gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a feature vector which can be used to detect objects like vehicles in image processing. The technique counts occurrences of gradient orientation in localized portions of an image. 

A HoG depends on three parameters:
- orientation: the number of oriented gradients that shall be taken into account
- pixels_per_cell: the number of pixels per cell
- cells_per_block: the number of cells per block

Here is an example using the `YCrCb` color space and different values for the HOG parameters:

| Color   |  Orienations |  PixelPerCell |  CellsPerBlock |
| ------- |:------------:|:-------------:|:--------------:| 
| `YCrCb` | [6,7,8,9]    | [8, 16]       | [2,4]          |


![][image2]

The code for this step is contained in`hog_features()` of the file `dataset_generator.py`. 

#### Color Histograms
I then explored different color spaces and color histogram as potential features. Color histgrams car computed in the `color_hist()` of the file `dataset_generator.py`
These are two examples of color histograms of randomly picked `car` and `notcar` images.

![][image3]

![][image3_2]

#### Spatial Binning
Finally, I also investigated the impact of spatial binning on the classifiers ability to detect vehicles.

![][image4]

![][image4_2]

### Training a Support Vector Machine
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

## Detecting Vechicles - Sliding Window Search

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

