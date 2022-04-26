#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install focal-loss


# In[2]:


pip install tensorflow-addons


# In[3]:


import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import glob
import os
import cv2
from tqdm import tqdm


# In[4]:


import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow import *

from tensorflow.keras.layers import *
from tensorflow.keras import *
import tensorflow as tf
from tensorflow import keras
#from  tensorflow.keras.initializers import *

import os
import sys
import random
import warnings
import cv2
import pandas as pd

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.morphology import disk
from skimage.filters import median
from focal_loss import BinaryFocalLoss
from tensorflow.keras.backend import *

import tensorflow_addons as tfa


# In[5]:


from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')


# In[6]:


image_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png")
image_name.sort()
#print(image_name)


# In[28]:


X_train = []

image_name = (glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png"))
image_name.sort()
images = []
for image in tqdm(image_name):
  img = cv2.imread(image,0)
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  img = cv2.bilateralFilter(img,9,75,75)
  X_train.append(img)


# In[29]:


X_train = np.array(X_train)
X_train = np.expand_dims(X_train,3)


# In[33]:


print(X_train.shape)

plt.imshow(np.reshape((X_train[1434]*255),(512,512)), cmap="gray")


# In[10]:


mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
#print(mask_name)


# In[35]:


Y_train = []

mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
images = []
for image in tqdm(mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_train.append(img)


# In[36]:


Y_train = np.array(Y_train)
Y_train = np.expand_dims(Y_train,3)


print(Y_train.shape)

plt.imshow(np.reshape((Y_train[1434]*255),(512,512)), cmap="gray")


# In[34]:


test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
#print(test_image_name)


# In[37]:


X_test = []

test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
test_images = []
for test_image in tqdm(test_image_name):
  img = cv2.imread(test_image,0)
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  img = cv2.medianBlur(img,5)
  X_test.append(img)


# In[38]:


X_test = np.array(X_test, dtype=np.float32)
X_test = np.expand_dims(X_test, 3)
print(X_test.shape)

plt.imshow(np.reshape((X_test[35]*255),(512,512)), cmap="gray")


# In[39]:


test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_mask_name.sort()
#print(test_image_name)


# In[40]:


Y_test = []

test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/mask/img/*png")
test_mask_name.sort()

for image in tqdm(test_mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_test.append(img)


# In[41]:


Y_test = np.array(Y_test, dtype=np.float32)
Y_test = np.expand_dims(Y_test, 3)
print(Y_test.shape)

plt.imshow(np.reshape((Y_test[35]*255),(512,512)), cmap="gray")


# In[50]:


def munet():
  
  input = keras.layers.Input((512,512,1))
  

  #encoder
  c = keras.layers.Conv2D(32 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(input)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(32 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  skip1 = c
  c = keras.layers.MaxPool2D((2,2),(2,2))(c)
  
  r = keras.layers.Conv2D(64 , (1,1) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  r = keras.layers.BatchNormalization()(r)
  r = keras.layers.Activation('relu')(r)
  
  c = keras.layers.Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Concatenate()([r,c])
  c = keras.layers.Conv2D(64, (1,1) , padding = 'same' , strides = 1 , activation = None)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  skip2 = c
  c = keras.layers.MaxPool2D((2,2),(2,2))(c)
  
  
  r = keras.layers.Conv2D(128 , (1,1) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  r = keras.layers.BatchNormalization()(r)
  r = keras.layers.Activation('relu')(r)
  
  c = keras.layers.Conv2D(128 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(128 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Concatenate()([r,c])
  c = keras.layers.Conv2D(128, (1,1) , padding = 'same' , strides = 1 , activation = None)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  skip3 = c
  c = keras.layers.MaxPool2D((2,2),(2,2))(c)
  
  
  r = keras.layers.Conv2D(256 , (1,1) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  r = keras.layers.BatchNormalization()(r)
  r = keras.layers.Activation('relu')(r)
  
  c = keras.layers.Conv2D(256 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(256 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Concatenate()([r,c])
  c = keras.layers.Conv2D(256, (1,1) , padding = 'same' , strides = 1 , activation = None)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  skip4 = c
  c = keras.layers.MaxPool2D((2,2),(2,2))(c)
  
  
  
  r = keras.layers.Conv2D(512 , (1,1) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  r = keras.layers.BatchNormalization()(r)
  r = keras.layers.Activation('relu')(r)
  
  c = keras.layers.Conv2D(512 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(512 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Concatenate()([r,c])
  c = keras.layers.Conv2D(512, (1,1) , padding = 'same' , strides = 1 , activation = None)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  skip5 = c
  c = keras.layers.MaxPool2D((2,2),(2,2))(c)
  
  
  
  #ASPP
  b0 = keras.layers.Conv2D(1024 , kernel_size = (1,1) , padding = 'same' , strides = (1,1))(c)
  b0 = keras.layers.BatchNormalization()(b0)
  b0 = keras.layers.Activation('relu')(b0)
  
  
  b1 = keras.layers.Conv2D(1024 , kernel_size = (3,3) , padding = 'same' , strides = (1,1) , dilation_rate = 1)(c)
  b1 = keras.layers.BatchNormalization()(b1)
  b1 = keras.layers.Activation('relu')(b1)
  
  b2 = keras.layers.Conv2D(1024 , kernel_size = (3,3) , padding = 'same' , strides = (1,1) , dilation_rate = 2)(c)
  b2 = keras.layers.BatchNormalization()(b2)
  b2 = keras.layers.Activation('relu')(b2)
  
  b3 = keras.layers.Conv2D(1024 , kernel_size = (3,3) , padding = 'same' , strides = (1,1) , dilation_rate = 3)(c)
  b3 = keras.layers.BatchNormalization()(b3)
  b3 = keras.layers.Activation('relu')(b3)
  
  #b4 = keras.layers.GlobalAveragePooling2D()(c)
  #b4 = keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
  #b4 = keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
  #b4 = keras.layers.Conv2D(512 , kernel_size = (1,1) , padding = 'same' , strides = (1,1))(b4)
  #b4 = keras.layers.BatchNormalization()(b4)
  #b4 = keras.layers.Activation('relu')(b4)
  #b4 = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize(x, (16,16),method='bilinear', align_corners=True))(b4)
  
  out = keras.layers.Concatenate()([b0,b1,b2,b3])
  out = keras.layers.Conv2D(1024 , kernel_size = (1,1) , padding = 'same')(out)
  out = keras.layers.BatchNormalization()(out)
  out = keras.layers.Activation('relu')(out)
  
  
  
  #decoder
  us = keras.layers.UpSampling2D((2,2))(out)   #upsampling with a pool of 2 by 2
  concat = keras.layers.Concatenate()([us, skip5]) # calculating the skip connection
  c = keras.layers.Conv2D(512 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(concat)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(512 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  
  us = keras.layers.UpSampling2D((2,2))(c)   #upsampling with a pool of 2 by 2
  concat = keras.layers.Concatenate()([us, skip4]) # calculating the skip connection
  c = keras.layers.Conv2D(256 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(concat)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(256 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  
  us = keras.layers.UpSampling2D((2,2))(c)   #upsampling with a pool of 2 by 2
  concat = keras.layers.Concatenate()([us, skip3]) # calculating the skip connection
  c = keras.layers.Conv2D(128 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(concat)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(128 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  
  us = keras.layers.UpSampling2D((2,2))(c)   #upsampling with a pool of 2 by 2
  concat = keras.layers.Concatenate()([us, skip2]) # calculating the skip connection
  c = keras.layers.Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(concat)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(64 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  
  us = keras.layers.UpSampling2D((2,2))(c)   #upsampling with a pool of 2 by 2
  concat = keras.layers.Concatenate()([us, skip1]) # calculating the skip connection
  c = keras.layers.Conv2D(32 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(concat)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  c = keras.layers.Conv2D(32 , (3,3) , padding = 'same' , strides = 1 , activation = None , dilation_rate = 1)(c)
  c = keras.layers.BatchNormalization()(c)
  c = keras.layers.Activation('relu')(c)
  
  
  output = keras.layers.Conv2D(1,(1,1),padding = "same" , activation = "sigmoid")(c)
  model = keras.models.Model(input,output)
  return model

model_munet = munet()
#adam = keras.optimizers.Adam(lr = 0.001)

from tensorflow.keras.backend import *
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = cast(y_true, 'float32')
    y_pred_f = cast(y_pred, 'float32')
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model_munet.compile(optimizer = "adam" , loss = BinaryFocalLoss(gamma=2) , metrics = ['acc'])
model_munet.summary()


# In[51]:


history = model_munet.fit(x=X_train,y=Y_train, batch_size=4,epochs=40,validation_data = (X_test,Y_test), callbacks=[callback])


# In[44]:


# Plot the model training accuracy on training dataset
print(history.history.keys())
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + history.history ['acc'], '--')
ax.plot([None] + history.history ['val_acc'], 'x-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train acc', 'Validation acc'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc') 


# In[45]:


# Plot the model training loss on training dataset
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + history.history['loss'], 'k--')
ax.plot([None] + history.history['val_loss'], '-')
# Plot legend and use the best location automatically: loc = 0.
ax.legend(['Train Loss', 'Validation Loss'], loc = 0)
ax.set_title('Training/Validation Loss per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')


# In[46]:


score = model_munet.evaluate(X_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[47]:


import matplotlib.pyplot as plt
import cv2 as cv
out_pred=[]
# X_test, Y_test = test()
out = model_munet.predict(X_test)
print(out.shape)
out=out>0.4

# For displaying image we have reshape size of predited and ground truth to size of 512 x 512
Out_image=np.reshape((out[65]*255),(512,512))
Y_Test_image =np.reshape((Y_test[65]*255),(512,512))

#path4 = '/content/drive/My Drive/TNBC/result_nc/'
#cv2.imwrite(path4 +'ncb4'+'.png',img)

plt.subplot(121)
plt.imshow(Out_image, cmap="gray")
plt.title("Predicted Image")
plt.subplot(122)
plt.imshow(Y_Test_image, cmap="gray")
plt.title("Ground truth Image")
plt.show()


# In[48]:


import numpy as np


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum() + 1)
    


# In[49]:


from statistics import mean
dice_scores = []
for m in range(155):
    diff= dice(Y_test[m],out[m])
    score= (diff)*100
    dice_scores.append(score)
#print("List of Dice Scores: ",dice_scores)
print("Mean of Test Images dice :",mean(dice_scores))


# In[ ]:


def jaccard_index(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

    


# In[ ]:


from statistics import mean
import tensorflow.keras.backend as K
print(out.shape)
print(Y_test.shape)
jaccard5 = []
for m in range(155):
    jc=jaccard_index(Y_test[m],out[m])
    jaccard5.append(jc)
#print("List of Jaccard Scores: ",jaccard5)
print("Mean of Test Jaccard Scores :",mean(jaccard5))


# In[ ]:


def recall_m(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true.astype(np.bool) * y_pred.astype(np.bool), 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true.astype(np.bool), 0, 1)))
        recall = true_positives / (possible_positives + 0.0001)
        return recall


# In[ ]:


from statistics import mean
import tensorflow.keras.backend as K
recall = []
for m in range(155):
    rc=recall_m(Y_test[m],out[m])
    recall.append(rc)
#print("List of recall: ",recall)
print("Mean of Test recall :",mean(recall))


# In[ ]:


def precision_m(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true.astype(np.bool) * y_pred.astype(np.bool), 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred.astype(np.bool), 0, 1)))
        precision = true_positives / (predicted_positives + 0.0001)
        return precision


# In[ ]:


from statistics import mean
import tensorflow.keras.backend as K
precision = []
for m in range(15):
    pc=precision_m(Y_test[m],out[m])
    precision.append(pc)
#print("List of precision: ",precision)
print("Mean of Test precision :",mean(precision))


# In[ ]:


def f1_m(y_true, y_pred):
    precision = precision_m(y_true.astype(np.bool), y_pred.astype(np.bool))
    recall = recall_m(y_true.astype(np.bool), y_pred.astype(np.bool))
    return 2*((precision*recall)/(precision+recall+0.00001))
    


# In[ ]:


from statistics import mean
import tensorflow.keras.backend as K
f1_score = []
for m in range(15):
    fs=f1_m(Y_test[m],out[m])
    f1_score.append(fs)
#print("List of f1_score: ",f1_score)
print("Mean of f1_score :",mean(f1_score))

