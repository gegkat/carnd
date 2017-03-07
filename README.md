#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/cnn-architecture-624x890.png "nvidia"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is based on the NVIDIA deep learning self driving car convolutional neural network architectue. 

Data is first normalized using a Keras lambda layer (model.py line 134).

Then the data is cropped using the Keras Cropping2D function to remove distracting informationa above the horizon and unnecessary information at the car hood. 

The data then goes through a series of 5 convolutions. These increase in depth from 24 to 64, have filter sizes of 5x5 or 3x3, and a subsamples of 2,2 or 1,1 (model.py lines 140-144).

The model includes RELU layers to introduce nonlinearity (model.py lines 140-144). 

After the convolutions, the neural has 4 fully connected layers of 100, 50, 10, and 1 nodes respectively. (model.py lines 145-149)


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 84, 119-120, 155-157). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 151).
  
####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of three laps of center lane driving and three extra iterations focusing on traversing the two sharpest turns of the track: the left turn with the dirt on the right and the right turn with the water on the left. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple architecture to develop and test my pipeline and then add sophistication until I had an architecture that neither underfit nor overfit. After finding a good architecture I adjusted my training data set until the car performed well on the test track. I continuously checking the fit of the model architecture as the training data set was adjusted, but ultimately the architecture needed no adjustment. The more difficult task was adjusting the training data to get a good result on the test track. 

My first step was to use a simple single fully connected layer. This helped me work out the bugs in my code from recording data through to testing in the simulator. This was fast enough to run on my personal laptop. 

Next I tried the lenet architecture and moved to a GPU on amazon web services. This architecture led to some success on the track, but was underfitting as evidenced by high mean squared error on the training data set. 

Finally I changed to the NVIDIA deep learning self driving car convolutional neural network architectue. This model had low training error and low validation error. I found that the validation error stopped decreasing after 3 epochs so I limited the training to 3 epochs to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were several problems when running the simulator even with low validation and training error. I addressed these by improving the training data set. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

I found that the keys to success in this project were to be disciplined and systematic when making adjustments and to focus on the training data, not the model architecture. I found that the NVIDIA model architecture performed well out of the box and needed not modifications. This architecture had no issue underfitting my data and I could avoid overfitting by training for only 3 epochs.  

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. This provided 4196 samples of steering data. When training the model on this data set the vehicle performed well on straight roads but failed to recover when it drifted to the side of the road and could not complete the first turn. 

For this training set and all subsequent ones, I preprocessed the images by normalizing all data between -1 and 1. I also cropped the images to ignore information above the horizon and below the car hood. Finally I shuffled the data and put 20% of the samples in the validation set. 

Next I added samples for the left and right cameras with a steering correction factor of 0.2. This tripled my data set. On this data set the vehicle made it to the bridge but had a bias of steering to the left.

To counteract the leftward bias, I added a mirror image of all of the images to the data set. Now I had 6 times the original data set with 25176 samples. Training on this data set corrected the bias but the vehicle failed to turn sharply enough on the two sharpest turns of the track. 

To finally achieve a complete run of the track I had to make two final adjustments. First I added more training data specifically of the two sharpest turns by recording driving of just these turns 3 times each, being careful to take the turns sharply. Second I tried to remove some bias towards straight driving by removing 40% of image samples with a steering angle less than 0.03. With this data set, the network learned to successfully complete the track. 