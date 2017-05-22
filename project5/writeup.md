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
[lane.py]: ./line.py
[parameters.p]: ./parameters.p

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/color_histogram.png
[image3_2]: ./output_images/color_histogram_notcar.png
[image4]: ./output_images/spatial_binning.png
[image4_2]: ./output_images/spatial_binning_notcar.png

[image5]: ./output_images/empty_regions.png
[image6]: ./output_images/hot_regions.png


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

## Usage
Shows all possible cmdline args:
```sh
PS> python .\pipeline.py -h
usage: python model.py [-d]|[-t datasetfile]|[-p modelfile]
arguments
-h                   Help
-d, --dataset        Generates the dataset dataset.tsv file.
-t, --train          Trains a svm model. Provide it with the dataset.tsv file. The output is a svm_tuned_dataset.pkl file
-p, --process        Process the video. Use svm_tuned_dataset.pkl as model file
```

Generate a new dataset:
```sh
PS> python .\pipeline.py -d
```

Train a model:
```sh
PS> python .\pipeline.py -t dataset.tsv
```

Process video:

```sh
PS> python .\pipeline.py -p svm_tuned_dataset.pkl
```

NOTE: You can download a pretrained model from [here](https://drive.google.com/file/d/0B2IrqAG6bsZWckZrNHFlS2VyQkE/view?usp=sharing).



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---

## Vehicle Detection Pipeline
In the following sections I will describe my pipeline that detects and visualizes vehicles and lane lines. The pipeline consists
of the following steps.

#### Preprocessing ####
1. Generate a dataset.
2. Train a classifier to detect vehicles

#### Image Processing Pipeline ####
1. Use a sliding window search to detect possible vehicles
2. Generate a heatmap. The heatmap is used to reduce false positives as well as to identify vehicles
3. Mark detected vehicles with a bounding box
4. Detect lane lines

## Preprocessing
Preprocessing includes dataset generation und training a classifier.

### 1. Dataset Generation
I downloaded the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) dataset which was provided by [Udacity](https://udacity.com) and copied them into two separate folders respectively. Then I iterated over each  `vehicle` and `non-vehicle` image and computed a feature vector. I implemented this step in the `create_feature_row()` function in the `dataset_generator.py` file. I also added the corresponding label/class (CAR=1, NOTCAR=0) to the end of the feature vector. Finally, I generated a [pandas](http://pandas.pydata.org/) dataframe from the feature vectors and stored the frame as csv file. This file was later used to generate training and test sets for the classifier.

Here is an example of a randomly chosen car and non-car image.

![][image1]

#### Histogram of Oriented Gradients (HOG)
The [histogram of oriented gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a feature vector which can be used to detect objects like vehicles in image processing. The technique counts occurrences of gradient orientation in localized portions of an image. 

A HOG depends on three parameters:
- orientation: the number of oriented gradients that shall be taken into account
- pixels_per_cell: the number of pixels per cell
- cells_per_block: the number of cells per block

Here is an example using the `YCrCb` color space and different values for the HOG parameters:

| Color   |  Orientations |  PixelPerCell |  CellsPerBlock |
| ------- |:------------:|:-------------:|:--------------:| 
| `YCrCb` | [6,7,8,9]    | [8, 16]       | [2,4]          |


![][image2]

The code for this step is contained in`hog_features()` of the file `dataset_generator.py`. 

#### Color Histograms
I then explored different color spaces and color histogram as potential features. Color histograms are computed in the `color_hist()` function of the file `dataset_generator.py`
These are two examples of color histograms of randomly picked `car` and `notcar` images.

![][image3]

![][image3_2]

#### Spatial Binning
Finally, I also investigated the impact of spatial binning on the classifiers ability to detect vehicles. Here are two examples. 

![][image4]

![][image4_2]

#### Feature Vector
My final feature vector consists of the following features:
- HOG features: For each channel
- Color histogram: For each channel
- Spatial binning

| Color   |  Orientation |  PixelPerCell |  CellsPerBlock | NBins | BinsRange |  Spatial BinningSize |
| ------- |:------------:|:-------------:|:--------------:|:-----:|:---------:|:----------------:| 
| `YCrCb` | 9            | 8             |  2             | 16    |  0 - 256  | 16x16            |

### 2. Training a Support Vector Machine
I trained a Support Vector Machine (SVM) as vehicle classifier/detector. Therefore, I loaded the previously generated dataset into memory, shuffled it and split it into a trainingset and testset. For the testset I used 20% of the total dataset. Trainingsets and testsets are normalized and then fed to the svm. I've also applied a GridSearch for hyper parameter tunings. After model evaluation I get the following metrics for it:

|  Metric   |  Value          |
|:---------:|:---------------:|
| Accuracy  | 0.995495495495 | 
| Precision | 0.99769186382  |
| Recall    | 0.993107409535 |
| F1 score  | 0.995394358089 |
 
The code can be found in the file `classifier.py` and the svm model is generated in the `train_models()` functions. The following lines are the most important ones: First I shuffle the dataset. Then I normalize the features. Therefore I use scikits `StandardScaler`. Then I create a training and a testset and fit the svm. Finally, I dump the model, the scaler, some metrices and the best_params to a file.
```python
        shuffled_dataset = shuffle(dataset.values)        
        shuffled_dataset = shuffle(shuffled_dataset)
        
        X = shuffled_dataset[:,0:num_features-1].astype(np.float64)        
        y = np.concatenate(shuffled_dataset[:,num_features-1:num_features].reshape(1,-1))
        
        # Normalize features
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
        
        if tune_model == True:
            parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            svr = svm.SVC()
            clf = GridSearchCV(svr, parameters)
        else:
            clf = svm.SVC( kernel='linear', C=1)
            
        clf.fit(X_train, y_train)
        
        ...
        
         dump(clf, X_scaler, metrics clf.best_params_, dumpfile)
```

## 3. Image Processing Pipeline
This is the part where vehicles are detected in each video frame.

#### Sliding Window Search
Sliding windows of different sizes are used to detect vehicles - actually the size of the sliding window is not really changed but search area is resized which has the same effect as changing the sliding window's size. This part is implemented in the `find_cars()` in the file `pipeline.py` and is based on the function that was provided in the Udacity lectures. The main idea is that we compute the HOGs only once and for the whole search area. Then we slide a window over this area extract the HOG features, the color histogram and spatial features, combine them to one feature vector and use the svm to predict a vehicle in this area. Potential areas are stored as rectangles (bounding boxes) and are returned to the caller for further processing. 
I used three different window size: 96x96, 128x128 and 160x160 respectively. 

#### Heatmap
The heatmap is used to detect vehicles as well as to filter false positives by using a certain threshold. The heatmap is the result of the accumulation of overlapping bounding boxes of several consecutive frames. The functions for heatmap generation and the final vehicle detections and visualization are `add_heat()`, `def apply_threshold()` and `def draw_labeled_bboxes()` - they can be found in the file `pipeline.py`. These functions were taken from the lectures. 
I set the threshold to 5 and bounding boxes of the last 10 frames are kept and used for thresholding.

The following images show the sliding windows at different scales and the image processing pipeline.

![][image5]

![][image6]

---


## Video Implementation

Here's a [link to my video result](./project_video_processed.mp4)



## Discussion
My pipeline is able to detect vehicles and suppresses successfully most false positives at least in the given test and project videos.
My pipeline may not work properly under different light or weather conditions - haven't test that yet. 
Its performance is also poor, especially the sliding window search takes a lot of time. This could be improved by:
1. Implement the pipeline in C/C++
2. Parallelization, e.g. one thread per window of a certain size.
3. Reduce the number of sliding windows by restricting the search area.
4. Use a different approach, e.g. semantic segmentation.

Manual feature engineering is also very time consuming - finding the right features needs lots of experimentation and takes time. This could be avoided by using e.g. a CNN. The advantages of such an approach:
1. No manual feature engineering needed.
2. Spatial information is taken into account.
3. End-to-end solution.
4. Maybe faster.

Finally, I think this project was again very interesting and funny. I'm looking forward to the next challenge.


