#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from google.colab.patches import cv2_imshow


# In[2]:


from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')


# In[3]:


image_name = glob.glob('/content/drive/MyDrive/breast_cancer/training/image/img/*png')
image_name.sort()
print(image_name)


# In[4]:


image_name = glob.glob('/content/drive/MyDrive/breast_cancer/training/image/img/*png')
image_name.sort()

listing = [cv2.imread(file) for file in image_name]




print(len(listing))


# In[14]:


plt.imshow(listing[1])


# In[9]:


data_dir = '/content/drive/MyDrive/breast_augmented_cancer/train/image/img1'
rows, cols,a = listing[0].shape
for i in range(0,len(listing)):
    rows,cols,a = listing[i].shape
    #print(rows,cols,a)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
    #N = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    image1 = cv2.warpAffine(listing[i],M,(cols,rows))
    #plt.imshow(image1)
   # image2 = cv2.warpAffine(listing[i],N,(cols,rows))
    cv2.imwrite(os.path.join(data_dir, str(i)+'.png' ), image1)
    #cv2.imwrite(os.path.join(data_dir, str((2*i) +97 )+'.png' ), image2)




# In[ ]:





# In[ ]:


for i in tqdm(range(0,len(listing))):
    rows,cols,a = listing[i].shape
    alpha = 1.5# Contrast control (1.0-3.0)
    beta = 1
    adjusted = cv2.convertScaleAbs(listing[i], alpha=alpha, beta=beta)
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
    # #N = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    # image1 = cv2.warpAffine(Z,M,(cols,rows))
   # image2 = cv2.warpAffine(listing[i],N,(cols,rows))
    cv2.imwrite(os.path.join(data_dir, str(i+ 820)+'.png' ), adjusted)
    #cv2.imwrite(os.path.join(data_dir, str((2*i) +97 )+'.png' ), image2)
    


# In[ ]:


listing1 = [cv2.imread(file) for file in tqdm(sorted(glob.glob("/content/drive/MyDrive/breast_augmented_cancer/train/image/img6/*.png"),key=os.path.getmtime))]

print(len(listing1))


# In[ ]:


data_dir = '/content/drive/MyDrive/breast_augmented_cancer/train/image/img5'
rows, cols,a = listing[0].shape
for i in tqdm(range(0,len(listing))):
    rows,cols,a = listing[i].shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
    #N = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    image1 = cv2.warpAffine(listing[i],M,(cols,rows))
   # image2 = cv2.warpAffine(listing[i],N,(cols,rows))
    cv2.imwrite(os.path.join(data_dir, str(i+ 195)+'.png' ), image1)
    #cv2.imwrite(os.path.join(data_dir, str((2*i) +97 )+'.png' ), image2)

    


# In[10]:


mask_name = glob.glob('/content/drive/MyDrive/breast_cancer/training/mask/img/*.png')
mask_name.sort()
print(mask_name)


# In[11]:


grlisting = [cv2.imread(file) for file in mask_name]


# In[13]:


plt.imshow(grlisting[1])


# In[17]:


data_dir = '/content/drive/MyDrive/breast_augmented_cancer/train/mask/img1'
for i in range(0,len(grlisting)):
    rows,cols,a = grlisting[i].shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
   # N = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    image1 = cv2.warpAffine(grlisting[i],M,(cols,rows))
   # image2 = cv2.warpAffine(grlisting[i],N,(cols,rows))
    cv2.imwrite(os.path.join(data_dir, str(i)+'.png' ), image1)
    #cv2.imwrite(os.path.join(data_dir, str((2*i) + 97)+'.tif' ), image2)
    


# In[ ]:




