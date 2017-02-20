### Finding Lane Lines on the Road

The first project of the self-driving car course deals with detecting lane lines on the road. The goals of this project are to correctly identify left and right lines on the road in video streams. 
Therefore an image processing pipeline shall be implemented which takes a video stream as input and outputs an annotated video stream showing the detected lines.

In this writeup I first present my pipeline, i.e. the image processing steps are carried out in order to successfully detect the lane lines in the provided videos. Finally, I will reflect my work and will provide pontential suggestions for further improvements.

---

**Image Processing Pipeline**
In this section I describe my image processing pipeline which shall be able to detect lane lines at least under the following assumptions and constraints:
* The lane lines to be detected are more or less straight.
* The position of the camera that takes images is always the same.
* Images to be processed have always the same shape.
* Lightning conditions in the image are always the same.



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
