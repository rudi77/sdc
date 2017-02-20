### Finding Lane Lines on the Road

The first project of the self-driving car course deals with detecting lane lines on the road. The goals of this project are to correctly identify left and right lines on the road in video streams. 
Therefore an image processing pipeline shall be implemented which takes a video stream as input and outputs an annotated video stream showing the detected lines.

In this writeup I first present my pipeline, i.e. the image processing steps that are carried out in order to successfully detect the lane lines in the provided videos. Finally, I will reflect my work and will provide pontential suggestions for further improvements.

---

**Tools, Development Environment**

Anaconda, jupyter and spyder.

**Libraries**

This project is written in python and uses the following libraries:
* [opencv](http://opencv.org/): An image processing library written c/c++ with bindings for python. It contains many usefull algorithms.
* [numpy](http://www.numpy.org/): A python package used for scientific computing.
* [matplotlib](http://matplotlib.org/): A plotting library.


**Image Processing Pipeline**

In this section I describe my image processing pipeline which shall be able to detect lane lines at least under the following assumptions and constraints:
* The lane lines to be detected are more or less straight.
* The position of the camera that takes images is always the same, i.e. lane lines are always at the same position relative to the camera.
* Lightening conditions in the image are always constant. Images with good contrast shall be provided.
* The provided video stream is processed image by image.

***The pipeline***

1. Read the image
[Input]: ./examples/input.png "Input"
2. Convert image to a grayscale

3. Apply gaussian blur to smooth the image.

4. Apply canny to detect edges.

4. Define an area of interest (AoI) and mask it with the image. Lane lines shall only be detected within the AoI.

5. Apply hough transform. Hough transform can be used to detect lines in an image.

6. Draw red lines over detected lane lines



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
