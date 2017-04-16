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
[writeup.md]: ./writeup.md

[//]: # (Literature References)
[adam]: https://arxiv.org/pdf/1412.6980.pdf
[optimizers]: http://sebastianruder.com/optimizing-gradient-descent/
[nvidia]: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

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
* [model.py][model.py] containing the script to create and train the model
* [data_generator.py][data_generator.py] containing a samples generator and several helper functions used to augment training data
* [drive.py][drive.py] for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md My report summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. 
```sh
python drive.py model.h5 [image_path]
```
The drive.py module takes as input the saved model h5 file and optionally an image path where images from the run will be stored. 

#### 3. Submission code is usable and readable

The [model.py][model.py] file contains the code for training and saving the convolution neural network. 
I have also added some useful command line options. The different options can be listed by calling the model.py with the help option.
```
python model.py -h

optional arguments
-h                   Help
-s, --summary        Model layer and parameter summary
-b, --batch_size=    Batch size
-p, --initial_epoch= Set the initial epoch. Useful when restoring a saved model.
-e, --epochs=        Number of epochs
-m, --model=         Load a stored model
-j, --model_to_json= Write architecture to a json file
```

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Training the model is done as follows.
```sh
python model.py -i ./data/driving_log.csv
```
Training and validation samples are generated in the ```split_samples(...)``` function. The trainings samples are read into a [pandas](http://pandas.pydata.org/) data frame. The samples are split into a training and validation set. I have not created an extra test set. The model's real performance will be tested with the simulator.
```python
def split_samples(trainingfiles, test_size=0.2):
    header = ["center","left", "right", "steering", "throttle", "brake", "speed"]
    dataframe = None
    
    for trainfile in trainingfiles:
        df = pd.read_csv(trainfile) if "data_base" in trainfile else pd.read_csv(trainfile, names=header)

        if dataframe is None:
            dataframe = df
        else:
            dataframe = dataframe.append(df)
    train_samples, validation_samples = train_test_split(dataframe, test_size=test_size)
    
    return train_samples, validation_samples
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
Finally, the model is created, configured and training is started. 
```python
  ...
  settings.model = create_model() if settings.model == None else settings.model  
  
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

The actual model is defined in the ```create_model()``` functions. Keras is used to create the convnet. First a __model__ is instantiated and then conv and fc layers are added. The model's architecture is explained in more detail in the next section. 
```python
def create_model():
    keep_prob = 0.7

    model = Sequential()
    
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x, name=None)))
    model.add(Lambda(lambda x: (x - 128.) / 128.))
    #model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200))))

    model.add(Conv2D(24, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(48, 5, strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))

    return model
```

The [data_generator.py][data_generator.py] is used to generate batches of data sets. The data generator is implemented in the ```generator(...)``` function. The function takes as input a list of all samples, the batch_size and a boolean isAugment which indicates whether augmented samples shall be derived from a real sample. The code snippet below shows one while loop which runs forever and an inner loop which iterates over generated samples chunks. The __yield__ keyword indicates that this function does not really return the requested result but a generator. 

[//]: # (https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/)

```python
def generator(samples, batch_size, isAugment = True):
    ...    
    while 1:
        samples = sklearn.utils.shuffle(samples)
        
        for batch_samples in chunker(samples,batch_size):
           
            images = []
            angles = []
            
            for batch_sample in chunker(batch_samples, 1):
                ...
                            
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield X_train, y_train
```
The data_generator.py module does also contain several other functions which are used for image augmentation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [nvidia's paper][nvidia]. The model is implemented in [model.py][model.py] in the ```create_model()``` function. The first layer crops the original image to 65x320 pixels.
```python
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
```
The second layer converts the color- to grayscale images. I use tensorflow's conversion method ```tf.image.rgb_to_grayscale```.
```python
model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x, name=None)))
```
Images are normalized in the third layer using one more Lambda.
```python
model.add(Lambda(lambda x: (x - 128.) / 128.))
```
The last Lambda resizes the images to 66x200 pixels.
```pyhton
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200))))
```


```
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 1)        0
_________________________________________________________________
lambda_2 (Lambda)            (None, 65, 320, 1)        0
_________________________________________________________________
lambda_3 (Lambda)            (None, 66, 200, 1)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        624
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dropout_3 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 251,019.0
Trainable params: 251,019.0
Non-trainable params: 0.0
_________________________________________________________________
```

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
