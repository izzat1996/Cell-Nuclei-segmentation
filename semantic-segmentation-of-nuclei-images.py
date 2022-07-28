#!/usr/bin/env python
# coding: utf-8

# - In this kernel I will be using **U-Net model** for segmentation of nuclei images.
# - U-Net model is commonly used in image segmentation problems. Links to get information about U-Net model - [link1](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47), [link2](https://en.wikipedia.org/wiki/U-Net)
# - Aim of this kernel is to provide all the steps clearly which are required for segmenting the nuclei images.
# 
# **Fork the kernel, make changes to this code and get your hands dirty :)**
# 
# ---
# 
# ## Load all libraries

# In[8]:


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("Loaded all the libraries")


# ## Data loading and data exploration

# In[9]:


train_fpath = "../input/stage1_train/"
test_fpath = "../input/stage1_test/"

print("No. of subfolders in train folder = ",len(os.listdir(train_fpath)))
print("No. of subfolders in test folder = ",len(os.listdir(test_fpath)))


# - Let's understand the directory structure of the train dataset and how is data organised so that we can load the data from the required folder only.

# In[10]:


print("Sub-folders in the first folder '{}':".format(os.listdir(train_fpath)[0]))
print(os.listdir(os.path.join(train_fpath, os.listdir(train_fpath)[0])))


# - So there are many subfolders in **../input/stage1_train/** folder which in turn contain **['masks', 'images']** subfolders.
# - The images and their corresponding masks have to be retrieved from **['masks', 'images']** subfolders.
# - Let's make a list of all subfolders in **../input/stage1_train/** so that we can retrieve images and corresponding images.

# In[11]:


#fetching train, test images ids
train_ids = next(os.walk(train_fpath))[1]
test_ids = next(os.walk(test_fpath))[1]

print("No. of train ids = ",len(train_ids))
print("No. of test ids = ",len(test_ids))


# - Now we have the train, test ids so let's load images and masks which will be useful for training in the succeeding steps.

# In[12]:


def load_image_data(fpath, ids):
    img_lst=[]
    for img_id in ids:
        #print(os.path.join(fpath, img_id, "images", img_id)+".png")
        img = cv2.imread(os.path.join(fpath, img_id, "images", img_id)+".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array = Image.fromarray(img, 'RGB')

        resized_img = img_array.resize((150, 150))

        img_lst.append(np.array(resized_img))
    return img_lst

#loading nuclei image data for training
X_train = load_image_data(os.path.join(train_fpath), train_ids)
print("No. of images loaded in X_train = ",len(X_train))
print(type(X_train))


# In[13]:


#loading nuclei image data for testing
X_test = load_image_data(os.path.join(test_fpath), test_ids)
print("No. of images loaded in X_test = ",len(X_test))
print(type(X_test))


# In[14]:


def load_masks_data(fpath, ids):
    img_lst=[]
    for img_id in ids:
        path = fpath + img_id
        for mask_file in os.listdir(path + '/masks/'):
            #print(path + '/masks/' + mask_file)
            img = cv2.imread(path + '/masks/' + mask_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_array = Image.fromarray(img, 'RGB')

            resized_img = img_array.resize((150, 150))

            img_lst.append(np.array(resized_img))
    return img_lst

#loading nuclei masks data for training
y_train = load_masks_data(os.path.join(train_fpath), train_ids)
print("No. of images loaded in y_train = ",len(y_train))
print(type(y_train))

