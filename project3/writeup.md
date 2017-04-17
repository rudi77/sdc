[//]: # (Image References)

[imgsim]: ./images/simulator.png "Simulator Image"
[convmodel]: ./images/model_architecture.png "My conv model"
[angledist]: ./images/distribution_orig_new_data.png "Steering angle distribution"
[centerline_image]: ./images/center_line_driving.png "Center line"
[curvesonly_image]: ./images/curve_driving.png "Curves only"
[recovery_image]: ./images/recovery_driving.png "Recovery mode"
[flipped_image]: ./images/flipped_image.png "Flipped images"
[brightness_changes]: ./images/brightness_changes.png "Brightness variations"
[cropped_grayscale_images]: ./images/cropped_grayscaled.png "Cropped and grayscaled"
[hw_setup_image]: ./images/hardwaresetup.jpg "Hardware setup"

[//]: # (File References)
[model.py]: ./model.py
[data_generator.py]: ./data_generator.py
[drive.py]: ./drive.py
[writeup.md]: ./writeup.md

[//]: # (Literature References)
[adam]: https://arxiv.org/pdf/1412.6980.pdf
[optimizers]: http://sebastianruder.com/optimizing-gradient-descent/
[nvidia]: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

## Behavioral Cloning

![][imgsim]

This is my third project of the sdc course. The overall objective of this project was to build a convolutional network that is
able to predict steering angles from images coming from Udactity's sdc simualtor. Finally we must show that our model is able to keep the car on track for one round on track one.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
  
---
### Files Submitted 

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [drive.py][drive.py] for driving the car in autonomous mode
* [model.py][model.py] containing the script to create and train the model
* [data_generator.py][data_generator.py] containing a samples generator and several helper functions used to augment training data
* model.h5 containing a trained convolution neural network 
* writeup.md My report summarizing the results

#### drive.py
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. 
```sh
python drive.py model.h5 [image_path]
```
The drive.py module takes as input the saved model h5 file and optionally an image path where images from the run will be stored. 

#### model.py
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
    ...
    train_samples, validation_samples = train_test_split(dataframe, test_size=test_size)
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

#### data_generator.py
The [data_generator.py][data_generator.py] is used to generate batches of data sets. The data generator is implemented in the ```generator(...)``` function. The function takes as input a list of all samples, the batch_size and a boolean isAugment which indicates whether augmented samples shall be derived from a real sample. The code snippet below shows one while loop which runs forever and an inner loop which iterates over generated samples chunks. The __yield__ keyword indicates that this function does not really return the requested result but a generator. Follow this [link](https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/) for a more detailed explanation on generators and the __yield__ keyword.

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

My final model is based on nvida's model which is descibed in this [paper][nvidia]. Convolutional and fully connected layers are the same. Dropout layers, which are missing in the original model (at least have not been mentioned in the paper), have been added to the fully connected layers. 
The model is implemented in [model.py][model.py] in the ```create_model()``` function. The next figure shows the layers that my model consists of: 
![][convmodel]
The first layer crops the original images to 65x320 pixels. The second layer converts the color images to grayscale. I used tensorflow's conversion method ```tf.image.rgb_to_grayscale```. Images are normalized in the third layer and resized to 66x200 pixels in the fourth layer. The resized images are fed into five convolutional layer with 24, 36, 48, 64 and 64 feature maps respectively. Moreover the first, second and third conv layer use 5x5 kernels whereas the last two conv layers use 3x3 kernels. The final conv layer is then flattened and used as the input for three fully connected layers. The last layer outputs a single value - the steering angle.

A nice summary of the layers can also be written to a console using the following command.
```sh
python.exe model.py -s
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

In order to prevent the model from overfitting I introduced dropout layers. After every fully connected layer a dropout layer was added but the last one, the output layer. A dropout rate of 0.2 is used. 
```python
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dropout(keep_prob))
    model.add(Dense(10))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
```
Furthermore, I've splitted the data set into a training and validation set. The training set was extended at runtime with further samples using different augmentation methods. The augmentation methods are described in detail below in another section.

#### 3. Model parameter tuning
- Optimizere: I used the [adam][adam] optimizer with its default settings.
- Loss functions: mean_squared_error is used as loss function. (Note: In the previous project we had to recognize different traffic signs which is a typical classification problem and therefore softmax can be used whereas this problem deals with the prediction of numerical values and therefore something like mean squared error has to be used.
- Dropout rate: 0.3 is used for dropout rate.

#### 4. Appropriate training data

I trained my model with data recorded from the first as well as the second track. I recorded several rounds of center line driving, recovering from off-track situations and curves only driving which means that I've only captured data when I was driving a left or right curve.

For details about how I created the training data, see the next section. 

### Solution Design Approach

#### 1. Finding an appropriate Model

My strategy for deriving a model architecture can be described as follows: 
I started to read and tried to understanding [nvidia's paper][nvidia] on their end-to-end solution for a self driving car. Nvidia had a working solution for my problem. So I decided to I implemented their model. In addition I added dropout layers to every fully connected layer to avoid overfitting.

Then I trained my model with the training set provided by Udacity. At the beginning the model performed not very well but after collecting more and more training data it performed better and better, especially after adding more recovey and curves only records.
Varying the brightness of the images also improved the model's capability to drive the car safely around the track.
After some trail and error and several hours behind my "WingMan" my model was finally able to drive the car around track "One" without leaving the road. 

#### 2. Creation of the Training Set & Training Process
Having good and lots of training samples is the meat and potatoes for all machine learning algorithms. This is even more true for deep networks. I downloaded the Udacity's [training set](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). The downloaded data.zip file contains a driving_log.csv file and an IMG folder full of recorded images from the car's cameras. Both, the csv file and the images are generated by the simulator.
The csv file consists of the following entries:
- center: Path to the center image
- left: Path to the left image
- right: Path to the right image
- steering: The steering angle for the center image. A value between -1.0 and 1.0.
- throttle: The throttle value
- brake: The brake value
- speed: The actual speed

For this project the center, left, right and steering values were taken into account. I started to train my model with these data only which did not lead to the desired result - my model was not able to keep the car on track but instead it directly guided the car into the lake. Well done model.
Looking at the steering value distribution of the provided training set indicates two things:
- The data set is small. There are about 8036 sample rows in the training_log.csv file
- The data set is unbalanced. Most steering values are distributed around 0.0. A steering value of 0.0 means that the car is driving straight ahead.

A model that is trained with a small, unbalanced data set could be biased and underfitted. Therefore, I started to collect more data. My final data set consists of 46632 samples - one samples comprises the left, center and right image. I tried to balance out my data set using different techniques. The original and final steering angle distributions are visualized in the following plot.

![alt  text][angledist]

My data set is still normal distributed around 0.0 but variance is greater than in the original data set.

##### Data collection Strategies
First things first. This is my setup at home that I've used for collecting data:-)
![alt text][hw_setup_image]
- Center line tracking:
To capture good driving behavior, I recorded several laps on track one and two using center lane driving. Here are example images of center lane driving:

![alt text][centerline_image]

- Curves only tracking:
To get more curve samples I recorded only when the steering angle was greater or less 0.0. These are example images from curve driving.

![alt text][curvesonly_image]

I added two additional recording modes to the simulator which simplified curves recording.
  - Mode _LeftCurves_: This recording mode captures only when the car drives a left curve. Enable or disable this mode by pressing L key.
  - Mode _RightCurves_: This recording mode captures only when the car drives a right curve. Enable or disable this mode by pressing T key.

- Recovery tracking:
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the model would learn to 
These images show what a recovery looks like starting from right side of the road.

![text alt][recovery_image]


##### Data Augmentation
My training set is augmented during training set batch generation. The functions for data augmentation are implemented in [data_generator.py][data_generator.py]. 
The following techniques are applied the images:
- Left and right camera images: Left and right camera images are used additionaly to the center image. Using a left and a right camera comes in handy when model must recover from being off-road. If we associate the image from the center camera with a certain left then we can we can map the image from the left camera with slightly softer left turn. Whereas the image from the right camera shall be associated with a harder left turn. Speaking in steering angles, a constant offset, 0.25, is added to the angle when the image from the left camera image is used. This offset is substracted from the angle when the image of right camera is used. The code for this operations is in the ```generator(...)``` function.
```python
  ...
  # Left image
  left_image, left_angle = image_and_angle(batch_sample, LEFT)
  images.append(left_image)
  angles.append(left_angle + angle_offset)

  # Right image
  right_image, right_angle = image_and_angle(batch_sample, RIGHT)
  images.append(right_image)
  angles.append(right_angle - angle_offset)
```

- Image flipping: Images whose corresponding angle is not 0.0 are flipped horizontally. This is done in this function.
```python
def flip_image(image, steering_angle):
    image_flipped = np.fliplr(image)
    flipped_angle = -steering_angle    
    return image_flipped, flipped_angle
```
This is an example of an flipped image.

![text alt][flipped_image]

- Brightness: The brightness of an image is randomly varied. An image is first converted into the YUV colorspace. Then every value of the y channel is multiplied with a certain factor. The factor is a value between 0.2 and 1.2 generated from uniform distribution generator. If the new value is greater then 255 then I set it to 255. Finally the image is converted back to the BGR color space and returned.
```python
def brightness(image, cfactor=0.5):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] =  np.where(img_yuv[:,:,0] * cfactor < 255, img_yuv[:,:,0] * cfactor, 255) 
            
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
```
Variations in the brightness of images is shown in the next example.

![text alt][brightness_changes]

#### Data Preprocessing
All images are finally cropped and converted to grayscale images before they are used by the first conv layer as input data. I convert them to grayscale, because my model is able to keep the car on track for at least one round on track one, the lake track. Using grayscale images reduced the number of parameters and also reduced training time. Both image cropping and grayscaling is done in Keras, implemented as a Lambda layer.
```python
  ...
  model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
  model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x, name=None)))
```
This is an example of an cropped and grayscaled image.
![text alt][cropped_grayscale_images]

#### Training Process
Collecting data, testing augmentation methods and experimenting with different model architectures was done locally on my laptop. Training the model was carried out on an EC2 instance with GPU support. The trained model was then tested again on my local machine.

### Conclusion
At the end my model is able to drive the car around track one. I haven't managed to improve my model such that it is also able to successfully drive the car on track two. In my mind I have to increase my data set. Improving data augmentation techniques should also have an impact on the model's performance.
