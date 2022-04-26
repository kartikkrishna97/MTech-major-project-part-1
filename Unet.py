#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install focal-loss


# In[5]:


pip install tensorflow-addons


# In[6]:


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
from tensorflow import *
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras import *
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras.initializers import *

import os
import sys
import random
import warnings
import cv2
import pandas as pd
#import tensorflow.get_collections

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.morphology import disk
from skimage.filters import median
from focal_loss import BinaryFocalLoss
import tensorflow_addons as tfa
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[7]:


from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')


# In[8]:


image_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png")
image_name.sort()
print(image_name)


# In[9]:


X_train = []

image_name = (glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png"))
image_name.sort()
images = []
for image in tqdm(image_name):
  img = cv2.imread(image,0)
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  img = cv2.medianBlur(img,5)
  X_train.append(img)


# In[10]:


X_train = np.array(X_train)
X_train = np.expand_dims(X_train,3)


# In[11]:


print(X_train.shape)

plt.imshow(np.reshape((X_train[1786]*255),(512,512)), cmap="gray")


# In[12]:


mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
print(mask_name)


# In[13]:


Y_train = []

mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
images = []
for image in tqdm(mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_train.append(img)


# In[14]:


Y_train = np.array(Y_train)
Y_train = np.expand_dims(Y_train,3)


print(Y_train.shape)

plt.imshow(np.reshape((Y_train[1786]*255),(512,512)), cmap="gray")


# In[15]:


test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
print(test_image_name)


# In[16]:


X_test = []

test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
test_images = []
for test_image in tqdm(test_image_name):
  img = cv2.imread(test_image,0)
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  img = cv2.medianBlur(img,5)
  X_test.append(img)


# In[17]:


X_test = np.array(X_test, dtype=np.float32)
X_test = np.expand_dims(X_test, 3)
print(X_test.shape)

plt.imshow(np.reshape((X_test[35]*255),(512,512)), cmap="gray")


# In[19]:


test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_mask_name.sort()
print(test_image_name)


# In[20]:


Y_test = []

test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/mask/img/*png")
test_mask_name.sort()

for image in tqdm(test_mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_test.append(img)


# In[21]:


Y_test = np.array(Y_test, dtype=np.float32)
Y_test = np.expand_dims(Y_test, 3)
print(Y_test.shape)

plt.imshow(np.reshape((Y_test[35]*255),(512,512)), cmap="gray")


# In[23]:


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Encode module
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = BatchNormalization() (c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
c1 = BatchNormalization() (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = BatchNormalization() (c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
c2 = BatchNormalization() (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = BatchNormalization() (c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
c3 = BatchNormalization() (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = BatchNormalization() (c4)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
c4 = BatchNormalization() (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = BatchNormalization() (c5)
c5 = Dropout(0.4) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
c5 = BatchNormalization() (c5)

# Decoder module
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = BatchNormalization() (u6)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = BatchNormalization() (c6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
c6 = BatchNormalization() (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = BatchNormalization() (u7)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = BatchNormalization() (c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
c7 = BatchNormalization() (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = BatchNormalization() (u8)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = BatchNormalization() (c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
c8 = BatchNormalization() (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = BatchNormalization() (u9)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = BatchNormalization() (c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
c9 = BatchNormalization() (c9)
outputs = Conv2D(1, 1, activation='sigmoid') (c9)
outputs = BatchNormalization()  (outputs)

model = Model(inputs=[inputs], outputs=[outputs])

# Summary of model created 
print(model.summary())


# In[25]:


history = model.fit(x=X_train,y=Y_train, batch_size=4,epochs=50,validation_data = (X_test,Y_test), callbacks=[callback])
#history = model.fit(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=125,epochs=65, callbacks=[callback])


# In[26]:


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


# In[27]:


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


# In[28]:


score = model_pp.evaluate(X_test, Y_test, batch_size=4)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[29]:


y_pred = model_pp.predict(X_test, batch_size=4)


# In[41]:


import matplotlib.pyplot as plt
import cv2 as cv
out_pred=[]
# X_test, Y_test = test()
out = model_pp.predict(X_test, batch_size=4)
print(out.shape)
out=out>0.4

# For displaying image we have reshape size of predited and ground truth to size of 512 x 512
Out_image=np.reshape((out[135]*255),(512,512))
Y_Test_image =np.reshape((Y_test[135]*255),(512,512))

#path4 = '/content/drive/My Drive/TNBC/result_nc/'
#cv2.imwrite(path4 +'ncb4'+'.png',img)

plt.subplot(121)
plt.imshow(Out_image, cmap="gray")
plt.title("Predicted Image")
plt.subplot(122)
plt.imshow(Y_Test_image, cmap="gray")
plt.title("Ground truth Image")
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
def precision(gt,mask):
  gt = gt.flatten()
  #print(gt.shape)
  mask = mask.flatten()
  #print(mask.shape)
  tn,fp,fn,tp = confusion_matrix(gt,mask,labels=[0,1]).ravel()
  prec = tp/(tp+fp+0.000001)
  return prec

####recall---
def recall(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask,labels=[0,1]).ravel()
  rec = tp/(tp+fn+0.0000001)
  return(rec)

###f1 score--

def f1_score(prec,rec):
  f1 = 2*(prec*rec)/(prec+rec)
  return f1

  ### jaccard 
def jaccard(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask,labels=[0,1]).ravel()
  rec = tp/(tp+fn+fp+0.000001)
  return(rec)

  ### jaccard 
def Overall(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  x = confusion_matrix(gt,mask,labels=[0,1]).ravel()
  y = np.array(x)
  tn = y[0]
  fp = y[1]
  fn = y[2]
  tp = y[3]
  
  rec = (tp+tn)/(tp+fp+fn+tn)
  return(rec)



sum = 0
for i in range(len(Y_test)):
  sum = sum + precision(Y_test[i],out[i])
prec = sum/len(Y_test)

sum = 0
for i in range(len(Y_test)):
  sum = sum + recall(Y_test[i],out[i])
rec = sum/len(Y_test)

sum = 0
for i in range(len(Y_test)):
  sum = sum + jaccard(Y_test[i],out[i])
jaccard1 = sum/len(Y_test)


sum = 0
for i in range(len(Y_test)):
  sum = sum + Overall(Y_test[i],out[i])
Overall1 = sum/len(Y_test)

f1 = f1_score(prec,rec)

print("Jaccard Index", jaccard1)
print("final f1", f1)
print("final precision",prec)
print("final recall",rec)
print("Overall Accuracy",Overall1)


# In[42]:


z = len(Y_test)
print(z)
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


# In[43]:



from statistics import mean
dice_scores = []
for m in range(155):
    diff= dice(Y_test[m],out[m])
    score= (diff)*100
    dice_scores.append(score)
#print("List of Dice Scores: ",dice_scores)
print("Mean of Test Images dice :",mean(dice_scores))


# In[44]:


def jaccard_index(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


# In[45]:


from statistics import mean
import tensorflow.keras.backend as K
jaccard5 = []
for m in range(155):
    jc=jaccard_index(Y_test[m],out[m])
    jaccard5.append(jc)
#print("List of Jaccard Scores: ",jaccard5)
print("Mean of Test Jaccard Scores :",mean(jaccard5))


# In[46]:


def recall_m(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true.astype(np.bool) * y_pred.astype(np.bool), 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true.astype(np.bool), 0, 1)))
        recall = true_positives / (possible_positives + 0.0001)
        return recall


# In[48]:


from statistics import mean
import tensorflow.keras.backend as K
recall = []
for m in range(155):
    rc=recall_m(Y_test[m],out[m])
    recall.append(rc)
#print("List of recall: ",recall)
print("Mean of Test recall :",mean(recall))


# In[49]:


def precision_m(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true.astype(np.bool) * y_pred.astype(np.bool), 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred.astype(np.bool), 0, 1)))
        precision = true_positives / (predicted_positives + 0.0001)
        return precision


# In[50]:


from statistics import mean
import tensorflow.keras.backend as K
precision = []
for m in range(155):
    pc=precision_m(Y_test[m],out[m])
    precision.append(pc)
#print("List of precision: ",precision)
print("Mean of Test precision :",mean(precision))


# In[51]:


def f1_m(y_true, y_pred):
    precision = precision_m(y_true.astype(np.bool), y_pred.astype(np.bool))
    recall = recall_m(y_true.astype(np.bool), y_pred.astype(np.bool))
    return 2*((precision*recall)/(precision+recall+0.00001))


# In[52]:


from statistics import mean
import tensorflow.keras.backend as K
f1_score = []
for m in range(155):
    fs=f1_m(Y_test[m],out[m])
    f1_score.append(fs)
#print("List of f1_score: ",f1_score)
print("Mean of f1_score :",mean(f1_score))

