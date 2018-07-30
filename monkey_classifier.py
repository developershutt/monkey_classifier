
# coding: utf-8

# In[57]:


import math
import numpy as np
from matplotlib import pyplot as plt
import scipy
import cv2
import glob
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from pathlib import Path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Dense , MaxPooling2D , Activation,Dropout,Flatten
import os

np.random.seed(1)


# In[58]:


#train_dir=Path('/home/manish/Desktop/training/')
#test_dir=Path('/home/manish/Desktop/validation/')

train_dir=Path('E:/training')
test_dir=Path('E:/validation')


# In[59]:


img=cv2.imread('E:/training/n0/n0030.jpg')
print(img.shape)
plt.imshow(img)


# In[60]:


cols=['Label','Latin Name','Common Name','Train Images','Validation Images']
labels=pd.read_csv('E:/monkey_labels.txt',names=cols,skiprows=1)
labels


# In[61]:


labels = labels['Common Name']
labels


# In[62]:


height=150
width=150
channels=3
seed=1337
batch_size=64
num_classes=10
epochs=20
data_argumentation=True
num_prediction=20

# Training generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(height,width), 
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')


# In[63]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[64]:


model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['acc'])
model.summary()


# In[68]:


history = model.fit_generator(train_generator,
                              steps_per_epoch= 1097 // batch_size,
                              epochs=epochs,
                              validation_data=train_generator,
                              validation_steps= 272 // batch_size)


# In[66]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()

