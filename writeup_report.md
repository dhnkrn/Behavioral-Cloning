#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_dk.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 video for car going around track 1 with the model

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_dk.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My goal was to start with a simple mode and make optimizations as required. The starting point to this approach was to model a convolution neural network based on LeNet-5. The input is fed to a Cropping Layer that removes the top 70 and bottom 30 pixels effectively removing the sky and dashboard in the image. This reduces the data size that the model trains on. A Lambda layer then normalizes the input values between -0.5 to 0.5.  The rest of the stack is a Lenet-5 based model comprising two convolution layers and three fully connected layers with RELU activations for non-linearity interleaved between them. The convolutions layers outputs are fed through MaxPooling layers after RELU.  The last fully connected layer has 1 output instead of 10, as in LeNet.

####2. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets by doing 80:20 split of the data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer with default learning rate.  I did not have to add dropout layers either.

####4. Appropriate training data
This was the hardest part of the project. The data collection employed 
1) driving at the center of the track
2) several recovery driving recordings
3) data augmentation in code

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to take a base LeNet-5 model used for Traffic Sign Classification and progressively make changes as required. The LeNet-5 model was particularly advantageous int that it is small enough for me train on CPU and as I discovered later sufficient to keep the car on road.

In order to gauge how well the training was working, I split my image and steering angle data into a training and  a validation set (80:20). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more recovering driving data and artificially augmented the data set by randomly flipping images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

<- 320x160x3 images
Cropping (70 top,30 bottom) -> 320x60x3 
Normalization -0.5 to +0.5 pixel values
Convolution2D(6@5x5 filters,activation="relu") with stride=1x1
MaxPooling2D with stride 2x2
Convolution2D(6@5x5,activation="relu") with stride=1x1
MaxPooling2D  with stride 2x2
Flatten output
Dense(120)
Dense(84)
Dense(1)
 -> steering angle output

####3. Creation of the Training Set & Training Process

I first recorded a lap on track one using center lane driving. Training the model with this data set and testing it on the first track showed several problems with my data set even though the validation accuracy was pretty good. The car did not really learn to drive on the center of the road and would veer to the side almost immediately after moving. The model basically had not seen images with the car off the center during training and did not how to handle those inputs during testing.

Visualizing the data set with the help of histograms of steering angles showed that the data was unbalanced. There way too many images with the car at the center and ~0 steering angles. To correct this, I removed 2/3 of such images.

The next problem was that the car move to one side of the road even if the road was straight. This was due to disproportional number of negative steering angles. I corrected this by flipping images randomly and adding them to the data set.

Although the car drove better, it still could not negotiate tight corners and road surface changes. This again pointed to not having sufficient data, I then added images from the left and right camera and included images from recovery driving.

I stopped training at 2 epochs beyond which validation accuracy was not improving. I used an adam optimizer so that manually training the learning rate wasn't necessary.
