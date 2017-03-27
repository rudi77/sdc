# Traffic Sign Recognition

This is the summary of the second project - classifying german traffic sign with a convolutional neural network that is built, trained and evaluated in [tensorflow](tensorflow.org).

---

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_images.png "Traffics Sign of the Training Set"
[image2]: ./examples/traffic_sign_samples_distributions.png "Sample Distributions"
[image3]: ./examples/training_set_samples_distribution.png "Training Set Sample Distribution"
[image4]: ./examples/augmented_trainset_distribution.png "Augmented training set distribution"
[image5]: ./examples/msnet_graph.png "Multiscale network"
[image6]: ./examples/train_vs_accuracy_130.png "Msnet Training vs validation accuracy"
[image12]: ./examples/lenet_train_vs_accuracy.png "LeNet Training vs validation accuracy"
[image13]: ./examples/lenet_train_vs_accuracy_0001_40_512_05.png "LeNet Training vs validation accuracy wiht dropout"


[image7]: ./examples/MyGermanTrafficSigns/1/speed_limit_30_1.jpg "Speed Limit 30 km/h"
[image8]: ./examples/MyGermanTrafficSigns/13/yield_1.jpg "Yield"
[image9]: ./examples/MyGermanTrafficSigns/14/stopSign1.jpg "Stop Sign"
[image10]: ./examples/MyGermanTrafficSigns/18/General_Caution_1.jpg "General Caution"
[image11]: ./examples/MyGermanTrafficSigns/40/roundabout_1.jpg "Roundabout"

[//]: # (Literature References)
[1]: http://www.people.usi.ch/mascij/data/papers/2011_ijcnn_committee.pdf
[2]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
I should also mention that I have implemented several functions used for image tranformation, visualization and others in separate python files. These files can be found here: [data_transformations.py](https://github.com/rudi77/sdc/blob/master/project2/data_transformations.py), [data_plotting.py](https://github.com/rudi77/sdc/blob/master/project2/data_plotting.py) and [data_helper.py](https://github.com/rudi77/sdc/blob/master/project2/data_helpers.py). I have imported them as modules into my [notebook](https://github.com/rudi77/sdc/blob/master/project2/Traffic_Sign_Classifier.ipynb) wherever they are being needed.

---

## Dataset Exploration

### 1. Dataset Summary
Here, I provide a summary of the german traffic sign data set. The pickled [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) has already been resized to 32x32 images and also has been split into training, validation and test sets. The code for this step is contained in the second code cell of the IPython notebook. For this task I used pyhton's built-in functions and numpy

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2. Exploratory Visualization

Different visualization techniques have been applied to the data sets. First I displayed one traffic sign per class of the
training set. Then I plotted the samples distribution of the training, validation and test sets. And finally I made a horizontal bar chart of the training set.
The code for this step is contained in the code cells 3,6, and 8 of the IPython notebook.  
Photographs of the traffic signs are taken under different lighting conditions and from different perspectives and distances.

![alt text][image1]

The sample distributions of the training, validation and test sets are shown in the next image. The distributions across the data sets look very similar and unbalanced. Some classes contain more then 1500 images whereas others contain less then 250 images. This could lead to biased predictions which means that some classes may be predicted more accurately than others. One way to overcome this problem is to increase the samples of the underrepresented data by artificially augmenting them.

![alt text][image2]

Finally I have also generated a horizontal bar chart showing the sample distribution of the training set with its class names instead of the class ids.

![alt text][image3]

---

## Design and Test a Model Architecture

### 1. Preprocessing
The following preprocessing steps are carried out before the model is trained. First the traffic signs were artificially augmented, then converted into grayscale images and normalized. Finally, the class ids are transformed into one hot encoded vectors.
- Augment training set: The bar charts above showed that data is unbalanced among the different traffic sign classes which could distort predictions. Therefore different geometric transformations like translation, rotation and contrast adaptation are applied on the existing training samples to augment the number of training examples per traffic sign class. This approach is based on data augmentation methods mentioned in [1][1] and [2][2]. The newly generated images are stored with their corresponding labels as a pickle file in a separate folder "./traffic-signs-data-augmented/augmented_training.p". Moreover, data generation is only executed once and only the training set will be augmented leaving the validation and test images untouched. The code for data augmentation is in code cell 6. For every traffic sign class I generate __n_augmented_samples = (max_samples - number_of_samples_of_class_n)__ so that in the end all classes are out-balanced as it is shown in the next bar chart below. 

![alt text][image4]

- Convert images to grayscale: I converted the images from color to grayscale mainly because I've this approach was also used in [[1][1]] and [[2][2]]. Converted images to grayscale may also reduce training time and memory usage.

- Normalize images: Grayscaled images are then normalized between -1 and 1 by subtracting 128 from each pixel and then dividing this value by 128. 

  px_new = (px_old - 128) / 128
  
  Input normalization is good practice - helps GDC to converge faster.

- Encode traffics sign classes into one hot vectors: Although labels were already one hot encoded in the template it should be mentioned as it is an important step and is a standard method in machine learning. This method transforms categorical data like the traffic sign classes into one-hot encoded vectors.

### 2. Model Architecture

The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

For this project I implemented a so-called multiscale network which is based on the convolutional network described in [[2][2]]. 
My network consists of the following layers:

| Layer         		| Layer name     |     Description	        					            | 
|:-----------------:|:--------------:|:----------------------------------------------:| 
| Input         		|                |  32x32x1 grayscale image   							      | 
| Convolution 5x5   |   conv1        |  1x1 stride, valid padding, outputs 28x28x38 	|
| RELU					    |			           |       									                        |
| Max pooling	      |   pool1        |  2x2 stride,  outputs 14x14x38 				        |
| Convolution 5x5	  |   conv2        |  1x1 stride, valid padding, outputs 10x10x64   |
| RELU					    |						     |               					                        |
| Max pooling	      |   pool2        |  2x2 stride,  outputs 5x5x64   				        |
| Fully connected		|   fc1          |  input 9048, output 500        							  |
| Fully connected		|   fc2          |  output 100                    							  |
| Fully connected		|   fc3          |  output 84                      							  |
| Fully connected		|   output       |  output 43                      							  |
| Softmax				    |   crossentropy |              									                |


It differs to traditional convnets in that it is not a strict feed forward network but instead it branches the output after the first pooling layer and feds it directly into the fully connected layer. Merging the output from different layers into the classifier provides different scales of receiptive fields to the classifier which should improve accuracy. The following code shows how one can merge the outputs of different layers into a single tensor. In my case I concatenated the output of "pool1" and "pool2" into a new tensor labeled as "fc0". This concatenation is also graphically presented in the next image below. First I flattened the outputs  and then I used tensorflow's ```tf.concat``` function which concatenates a list of tensors values along a certain dimension axis.
```python
    with tf.name_scope("fc0"):    
        # Flatten Input = 14x14x38. Output = 7448
        conv1_flattened = flatten(conv1)
        # Flatten. Input = 5x5x64. Output = 1600    
        conv2_flattened = flatten(conv2)
        fc0 = tf.concat([conv1_flattened, conv2_flattened], 1)
```

For further details please refer to [[2][2]]  It is shown below in the next image.

![alt text][image5]

### 3. Model Training
At the beginning of this project I executed my models on my laptop's CPU which is an Intel I7 with four cores but later on I executed all my experiments on a "g2.2xlarge" EC2 instance which drastically reduced execution time and therefore, I was able to try different network architectures and hyperparameters. Finally I used the following settings and hyperparameters:

* Number of training examples: 91589 - the training set comprises the orginal training set plus my augmented set.
* Optimizer : AdamOptimizer. This one was taken from the LeNet example from lesson 9. I haven't tried others.
* Batch size: 1024. I've also tried smaller batch sizes like 128, 256 and 512 but finally chosen 1024 because the GPU was able to handle it. 
* Epochs: 130. I stopped training when I reached a training accuracy of 99% or more.
* Learning rate: 0.001. I've also tried different learning rates.
* Dropout probability: 0.5. In my final network I used dropouts in every fully connected layer to overcome overfitting.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ...

### 4. Solution Approach

I started with a modified LeNet model and trained it with the provided training set without any additional (artificially augmented) training data. With this approach I reached a very high accuracy on the training set - over 98% but was relatively low on the validation set. This discrepancy between training and validation set is shown in the next plot.  

![alt text][image12]

I tried to improve validation accuracy by introducing dropouts into the fully connected layer of my model. Added dropouts to the network can be easily done in tensorflow by using tensorflow's built in dropout function. The code below, which is located in code cell 8, shows how dropout can be integrated into a fully connected layer.

```python
def fc_layer(x, w_shape, b_size, layername, mu = 0.0, sigma = 0.1, keep_prob = 1.0):
    with tf.name_scope(layername):
        with tf.name_scope("weights"):
            fc_W = tf.Variable(tf.truncated_normal(shape=w_shape, mean = mu, stddev = sigma))
            variable_summaries(fc_W)
        with tf.name_scope("biases"):
            fc_b = tf.Variable(tf.zeros(b_size))
            variable_summaries(fc_b)
        fc = tf.matmul(x, fc_W) + fc_b
        fc = tf.nn.relu(fc)
        fc = tf.nn.dropout(fc, keep_prob)
        
        return fc
```        
Adding dropouts with probability of 0.5 improved my model's performance:
* Training Accuracy: 99.3 %
* Validation Accuracy: 94.5 %
* Test Accuracy: 92.4 %

The corresponding accuracy plot is presented in the next plot.

![alt text][image13]

After reading the proposed paper [[2][2]] by Pierre Sermanet and Yann LeCun I decided to implement my own multiscale-net based on their work. I liked the idea of being able to detect traffic signs on different scales by using such an architecture - as it is in real world a traffic sign in front will become larger and larger when heading towards it. Furthermore, I've have augmented my training set to further improve my model's performance. I have already described in detail my data augmentation approach in detail. And again this approach improved my model's accuracy.

My final model results were:
* training set accuracy of 99.1 %
* validation set accuracy of 95.9 %
* test set accuracy of 94.9 %

The following plot shows the training and validation accuracies over 130 Epochs.

![alt text][image6]


The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.
 
---

### Test a Model on New Images

#### 1. Acquiring New Images
####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

The first image might be difficult to classify because ...

#### 2. Performance on New Images
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Model Certainty - Softmax Probabilities
####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
