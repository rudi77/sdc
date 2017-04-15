[//]: # (Image References)

[imgsim]: ./images/simulator.png "Simulator Image"
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[//]: # (File References)
[model.py]: ./model.py
[data_generator.py]: ./data_generator.py
[drive.py]: ./drive.py

[//]: # (Literature References)
[adam]: https://arxiv.org/pdf/1412.6980.pdf
[optimizers]: http://sebastianruder.com/optimizing-gradient-descent/

### Under Construction...

![][imgsim]

### Behavioral Cloning

This is my third project of the sdc course. The overall objective of this project was to build a convolutional network that is
able to predict steering angles from images coming from Udactity's sdc simualtor. Finally we must show that our model is able to keep the car on track for one round on the lake track.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* data_generator.py containing a samples generator and several helper functions used to augment training data
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md My report summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. This file has not been adapted by me.
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py][model.py] file contains the code for training and saving the convolution neural network. 
I have also added some useful command line options. The different options can be listed by calling the model.py with the help option.
```
python model.py -h
```
```
usage: python model.py -i training_log.csv
optional arguments
-h                   Help
-s, --summary=       Model layer and parameter summary for a certain model, either 'nvidia' or 'rudi'
-a, --arch=          The model that shall be used. Either 'nvidia' or 'rudi'. Default is 'nvidia'
-b, --batch_size=    Batch size
-e, --epochs=        Number of epochs
-m, --model=         Load a stored model
-j, --model_to_json= Write architecture to a json file
-p, --initial_epoch= Set the initial epoch. Useful when restoring a saved model
```

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Training the model is done as follows.
```sh
python model.py -i ./data/driving_log.csv
```
The trainings samples are read into a [pandas](http://pandas.pydata.org/) data frame. The samples are split into a training and validation set using a split value of 0.2, i.e. 80% will be used for training and 20% for model validation. I have not created an extra test set. The model's real performance will be tested with the simulator.
```python
  ...
  tf_frames = [pd.read_csv(trainfile) for trainfile in settings.trainingfiles]
  # merge all frames into one
  dataframe = pd.concat(tf_frames)
  # split samples into training and validation samples.
  train_samples, validation_samples = train_test_split(dataframe, test_size=0.2)
```
Then I create two data generators - one for the training data and one for the validation data. Keras uses these generators to retrieve data for fitting and validating the model. The data generator is implemented in the [data_generator.py][data_generator.py] file. The training generator provides the training samples as well as augmented samples which are generated at runtime. Data augmentation is explained in detail in a subsequent section.
```python
  ...
  train_generator = dg.generator(train_samples, batch_size=settings.batches)
  validation_generator = dg.generator(validation_samples, batch_size=settings.batches, isAugment=False)
```
After that I instantiate Kera's ModelCheckpoint class. It can be used to save the model's after every epoch or when a certain condition is met, e.g. current validation loss is smaller then the previous one. The ModelCheckpoint instance must be registered at Kera's callback system, otherwise checkpointing won't be carried out. Therefore I add the ModelCheckpoint instance to my callbacks_list which is passed to the model's fit_generator method.
```python
  ...
  filepath="checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [callback_tb, checkpoint]
```
Finally, the model is configured and training is started. I used the [adam][adam] optimizer with its default settings and as a loss function mean_squared_error is used. (Note: In the previous project we had to recognize different traffic signs which is a typical classification problem and therefore softmax can be used whereas this problem deals with the prediction of numerical values and therefore something like mean squared error has to be used.)
```python
  ...
  settings.model.compile(optimizer="adam", loss='mse')

  print("start training")
  settings.model.fit_generator(train_generator,
                        steps_per_epoch = len(train_samples) / settings.batches,
                        epochs = settings.epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples) / settings.batches,
                        callbacks=callbacks_list,
                        initial_epoch=settings.initial_epoch)
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
