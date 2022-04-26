#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from tensorflow.keras.backend import *


# In[ ]:


pip install focal-loss


# In[ ]:


pip install tensorflow-addons


# In[ ]:


from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')


# In[ ]:


image_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png")
image_name.sort()
#print(image_name)


# In[ ]:


X_train = []

image_name = (glob.glob("/content/drive/MyDrive/breast_cancer1/training/image/img/*png"))
image_name.sort()
images = []
for image in tqdm(image_name):
  img = cv2.imread(image,0)
  x = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  y = cv2.bilateralFilter(x,9,75,75)
  X_train.append(y)


# In[ ]:


X_train = np.array(X_train)

#print(X_train.shape)
X_train = np.expand_dims(X_train,3)


# In[ ]:


print(X_train.shape)


# In[ ]:


plt.imshow(np.reshape((X_train[20]*255),(512,512)), cmap="gray")


# In[ ]:


mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
#print(mask_name)


# In[ ]:


Y_train = []

mask_name = glob.glob("/content/drive/MyDrive/breast_cancer1/training/mask/img/*png")
mask_name.sort()
images = []
for image in tqdm(mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_train.append(img)


# In[ ]:


Y_train = np.array(Y_train)
Y_train = np.expand_dims(Y_train,3)


print(Y_train.shape)


# In[ ]:


Y_train = np.array(Y_train)
Y_train = np.expand_dims(Y_train,3)


print(Y_train.shape)

plt.imshow(np.reshape((Y_train[20]*255),(512,512)), cmap="gray")


# In[ ]:


test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
#print(test_image_name)



# In[ ]:


X_test = []

test_image_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_image_name.sort()
test_images = []
for test_image in tqdm(test_image_name):
  img = cv2.imread(test_image,0)
  x = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  y = cv2.bilateralFilter(x,9,75,75)
  X_test.append(y)


# In[ ]:


X_test = np.array(X_test, dtype=np.float32)
X_test = np.expand_dims(X_test, 3)
print(X_test.shape)

plt.imshow(np.reshape((X_test[35]*255),(512,512)), cmap="gray")



# In[ ]:


test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/image/img/*png")
test_mask_name.sort()
#print(test_image_name)


# In[ ]:


Y_test = []

test_mask_name = glob.glob("/content/drive/MyDrive/breast_cancer/test/mask/img/*png")
test_mask_name.sort()

for image in tqdm(test_mask_name):
  img = cv2.imread(image,0)/255
  img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
  #img = cv2.medianBlur(img,5)
  Y_test.append(img)


# In[ ]:


Y_test = np.array(Y_test, dtype=np.float32)
Y_test = np.expand_dims(Y_test, 3)
print(Y_test.shape)

plt.imshow(np.reshape((Y_test[35]*255),(512,512)), cmap="gray")


# In[ ]:


# from tensorflow.keras.backend import *



class Loaded_Model:

    def __init__(self):
        pass  

    def add_conv_layer(self, prev_layer, num_filters, kernel_size, dilation_rate, dropout_rate = 0.1, activation = 'relu',kernel_initializer='he_normal', padding = 'same'):
        conv1 = Conv2D(num_filters, kernel_size, dilation_rate = dilation_rate, activation = activation,kernel_initializer=kernel_initializer, padding = padding)(prev_layer)
        u1 = BatchNormalization()(conv1)
        drop_n = Dropout(dropout_rate)(u1)
        conv_n = Conv2D(num_filters, kernel_size, activation = activation,kernel_initializer=kernel_initializer, padding = padding)(drop_n)
        u2 = BatchNormalization()(conv_n)
        return u2

    def add_dense(self, prev_layer, num_filters, kernel_size, activation = 'relu',kernel_initializer='he_normal', padding = 'same'):
        conv1 = Conv2D(num_filters, kernel_size, activation = activation,kernel_initializer=kernel_initializer, padding = padding)(prev_layer)
        u3 = BatchNormalization()(conv1)
        concat1 = concatenate([prev_layer, u3])
        conv2 = Conv2D(num_filters, kernel_size, activation = activation,kernel_initializer=kernel_initializer, padding = padding)(u3)
        u4 = BatchNormalization()(conv2)
        concat2 = concatenate([u4,concat1])
        conv3 = Conv2D(num_filters, kernel_size, activation = activation,kernel_initializer=kernel_initializer, padding = padding)(concat2)
        u5 = BatchNormalization()(conv3)
        return u5

    def add_dilated(self, prev_layer, num_filters, kernel_size, dilation_rate, activation = 'relu', kernel_initializer='he_normal', padding = 'same'):
        conv1 = Conv2D(num_filters, kernel_size, dilation_rate = dilation_rate, activation = activation, kernel_initializer=kernel_initializer, padding = padding)(prev_layer)
        u6 = BatchNormalization()(conv1)
        conv2 = Conv2D(num_filters, kernel_size, dilation_rate = dilation_rate, activation = activation, kernel_initializer= kernel_initializer, padding = padding)(u6)
        u7 = BatchNormalization()(conv2)
        concat = add([prev_layer, u7])
        return concat

    def add_High_RP(self, prev_layer, num_filters, kernel_size, dilation_rate1, dilation_rate2, activation = 'relu',kernel_initializer='he_normal', padding = 'same'):
        dilated1_1 = self.add_dilated(prev_layer, num_filters, kernel_size, 1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated1_2 = self.add_dilated(dilated1_1, num_filters, kernel_size, 1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated1_3 = self.add_dilated(dilated1_2, num_filters, kernel_size, 1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)

        concat1 = concatenate([prev_layer, dilated1_3])

        dilated2_1 = self.add_dilated(dilated1_3, num_filters, kernel_size, dilation_rate1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated2_2 = self.add_dilated(dilated2_1, num_filters, kernel_size, dilation_rate1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated2_3 = self.add_dilated(dilated2_2, num_filters, kernel_size, dilation_rate1, activation = activation,kernel_initializer=kernel_initializer, padding = padding)

        concat2 = concatenate([concat1, dilated2_3])

        dilated3_1 = self.add_dilated(dilated2_3, num_filters, kernel_size, dilation_rate2, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated3_2 = self.add_dilated(dilated3_1, num_filters, kernel_size, dilation_rate2, activation = activation,kernel_initializer=kernel_initializer, padding = padding)
        dilated3_3 = self.add_dilated(dilated3_2, num_filters, kernel_size, dilation_rate2, activation = activation,kernel_initializer=kernel_initializer, padding = padding)

        concat3 = concatenate([concat2, dilated3_3])
        return concat3

    def add_conv_transpose_layer(self, prev_layer, num_filters, kernel_size, strides, padding = 'same',kernel_initializer='he_normal', activation = 'relu'):
        upsampler = Conv2DTranspose(num_filters, kernel_size, strides = strides, padding = padding,kernel_initializer=kernel_initializer, activation = activation)(prev_layer)
        return upsampler

   
    def add_concat(self, prev_layer1, prev_layer2):
        concatenated = concatenate([prev_layer1, prev_layer2])
        return concatenated

    def add_mpool(self, prev_layer, pool_size = (2, 2)):
        mpool = MaxPooling2D(pool_size)(prev_layer)
        return mpool

    def add_final_layer(self, prev_layer, activation = 'sigmoid', kernel_size = (1, 1)):
        self.out = Conv2D(1, kernel_size, activation = activation)(prev_layer)
    
    def get_model(self, shape):
        
        self.inp = Input(shape = shape)

        dense1_1 = self.add_dense(self.inp, 16, (3,3))
        down_12 = self.add_mpool(dense1_1, pool_size = (2,2))

        dense1_2 = self.add_dense(down_12, 64, (3,3))
        down_23 = self.add_mpool(dense1_2, pool_size = (2,2))

        dense_bottom = self.add_dense(down_23, 128, (3,3))

        up1_32 = self.add_conv_transpose_layer(dense_bottom, 64, (2,2), 2)
        up2_32 = self.add_conv_transpose_layer(dense_bottom, 64, (4,4), 4)
        skip2 = self.add_concat(dense1_2, up1_32)
        dense2_2 = self.add_dense(skip2, 64, (3,3))
        hrp2 = self.add_High_RP(down_12, 16, (3,3), 2, 3)
        hrp_concat2 = self.add_concat(dense2_2, hrp2)

        up1_21 = self.add_conv_transpose_layer(hrp_concat2, 16, (2,2), 2)
        skip1 = self.add_concat(dense1_1, up1_21)
        skip1 = self.add_concat(skip1, up2_32)
        dense2_1 = self.add_dense(skip1, 16, (3,3))
        hrp1 = self.add_High_RP(dense1_1, 16, (3,3), 2, 3)
        hrp_concat1 = self.add_concat(dense2_1, hrp1)

        conv2 = self.add_conv_layer(hrp_concat1, 8, (3,3), 1)
        conv3 = self.add_conv_layer(conv2, 1, (3,3), 1)
        self.add_final_layer(conv3)
        
        
        

        loss_iou = tfa.losses.sigmoid_focal_crossentropy
        def dice_coef(y_true, y_pred, smooth=1):
          y_true_f = cast(y_true, 'float32')
          y_pred_f = cast(y_pred, 'float32')
          intersection = sum(y_true_f * y_pred_f)
          return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
        
        def dice_coef_loss(y_true, y_pred):
            return 1-dice_coef(y_true, y_pred)
        def tversky(y_true, y_pred, smooth=1, alpha=0.7):

          y_true_pos = flatten(y_true)
          y_pred_pos = flatten(y_pred)
          true_pos = sum(y_true_pos * y_pred_pos)
          false_neg = sum(y_true_pos * (1 - y_pred_pos))
          false_pos = sum((1 - y_true_pos) * y_pred_pos)
          return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
          
        def tversky_loss(y_true, y_pred):
            return 1 - tversky(y_true, y_pred)
        def focal_tversky_loss(y_true, y_pred, gamma=0.75):
            tv = tversky(y_true, y_pred)
            return pow((1 - tv), gamma)
        
        model = Model(inputs = self.inp, outputs = self.out)
        # adam = Adam(lr=1*1e-4, decay=5*1e-8)
        # model.compile(optimizer='adam', loss=loss_iou, metrics = ['acc'])
        # def step_decay(epoch):
        #   initial_lrate = 0.001
        #   drop = 0.5
        #   epochs_drop = 10.0
        #   lrate = initial_lrate * math.pow(drop,  
        #    math.floor((1+epoch)/epochs_drop))
        #   return lrate
        # callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
        adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_iou = tfa.losses.sigmoid_focal_crossentropy
        model.compile(optimizer='adam',loss = dice_coef_loss , metrics = ['acc'])
        return model
      


# In[ ]:


# m = Loaded_Model()
# model_new = m.get_model(shape=(512,512,1))
# model_new.summary()

m = Loaded_Model()
model_new = m.get_model(shape=(512,512,1))
model_new.summary()



# In[ ]:


def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

history = model_new.fit(x=X_train,y=Y_train, batch_size=4,epochs=40,validation_data = (X_test,Y_test), callbacks=[callback])


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# # y_pred = cnn.predict(X_test)
score = model_new.evaluate(X_test, Y_test,batch_size=4)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


y_pred = model_new.predict(X_test,batch_size=4)


# In[ ]:


import matplotlib.pyplot as plt
import cv2 as cv
out_pred=[]
# X_test, Y_test = test()
out = model_new.predict(X_test,batch_size=4)
print(out.shape)
out=out>0.4
print(out.shape)

# For displaying image we have reshape size of predited and ground truth to size of 512 x 512
Out_image=np.reshape((out[97]*255),(512,512))
Y_Test_image =np.reshape((Y_test[97]*255),(512,512))

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
    




# In[ ]:


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

