
# coding: utf-8

# In[7]:

import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

get_ipython().magic('matplotlib inline')
images = []
steering_angles = []


# In[8]:

images_orig = []
steering_angles_orig = []

for d in ["driving_data_1", "driving_data_2", "driving_data_3","driving_data_4","driving_data_5"] :
    csvfile = open("./"+d+"/driving_log.csv")
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[3])
        
        #filter out 2/3 of very small steering angles
        if angle > -0.001 and angle < 0.001:
            if np.random.randint(0,3) is not 0:
                continue
                
        im = mpimg.imread(line[0].strip())
        images_orig.append(im)
        steering_angles_orig.append(angle)
                   
        #don't use left and right camera images for small steering angle images
        if angle > -0.001 and angle < 0.001:
            continue 
        
        im = mpimg.imread(line[1].strip())
        images_orig.append(im)
        steering_angles_orig.append(angle - 0.25)
            
        im = mpimg.imread(line[2].strip())
        images_orig.append(im)
        steering_angles_orig.append(angle + 0.25)


train_images, validation_images, train_angles, validation_angles = train_test_split(images_orig,
                                                                                    steering_angles_orig,
                                                                                    test_size=0.2, random_state=42)


# In[2]:

def data_generator(images, angles, batch_size=32):
    size = len(images)
    while(1):
        shuffle(images, angles, random_state=42)
        for batch_start in range(0, size, batch_size):
                batch_images = []
                batch_angles = []
                
                batch_end = batch_start + batch_size
                if (batch_end) > size: #batch_end may be greater than the data set size
                    batch_end = size
    
                for i in range(batch_start, batch_end):
                    im = images[i]
                    an = angles[i]
                    
                    #flip images randomly 2/3 of the time
                    if np.random.randint(0,3) is not 0:
                        im = np.flip(im, 1)
                        an *= -1.0
        
                    batch_images.append(im)
                    batch_angles.append(an)
                
                yield np.array(batch_images), np.array(batch_angles)


# In[3]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

batch_size = 32
train_generator = data_generator(train_images, train_angles, batch_size)
validation_generator = data_generator(validation_images, validation_angles, batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((70,30), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(60,320,3)))
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_images)/batch_size,
                    validation_data = validation_generator, validation_steps=len(validation_images)/batch_size,
                    epochs=4)

model.save('model_generator.h5')


# In[ ]:



