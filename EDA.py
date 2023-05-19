#!/usr/bin/env python
# coding: utf-8

# # gold miner
# 
# ### Group Members:  Cheng Nie, Zhenxuan Liang, Jiayang Xu, Yijun Xie

# # Links:

# ### Kaggle Competition: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
# 
# ### Dataset Downloading: https://drive.google.com/drive/folders/1RqifyoVkT242wa4k5BqbHHjZmSvqlLr6?usp=share_link

# # Import all the packages

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import seaborn as sns
import plotly.graph_objects as go
import plotly_express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import re
import os 
import json
import glob
import random
import shutil
import itertools
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from pathlib import Path
import cv2
from PIL import Image 
import skimage.io as io
from skimage import data_dir,io,color
import keras
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc
import tensorflow as tf
from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, InceptionResNetV2, MobileNetV2 ,EfficientNetB7,EfficientNetB5,EfficientNetB3
from tensorflow.keras.layers import Dense, Activation,Flatten,Conv2D,Dropout,MaxPooling2D,AveragePooling2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers.experimental import preprocessing
from keras.utils.vis_utils import plot_model
import h5py
from keras import Model
from sklearn.manifold import TSNE
import warnings 
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis

# In[56]:


# Read the csv file in Python 
Train_Classification = pd.read_csv('./Cassava leaf/train.csv')
Train_Classification = shuffle(Train_Classification,random_state = 9417)
Train_Classification['label'] = pd.Series(Train_Classification['label'], dtype="string")
Train_Classification


# In[57]:


# open the json file in Python
disease_names = open('./Cassava leaf/label_num_to_disease_map.json')
disease_names = json.load(disease_names)
disease_names


# In[58]:


# Map the diseases name with the labels
Train_Classification["Classification"] = Train_Classification["label"].apply(lambda x: disease_names.get(x))
Train_Classification


# ##  Data Distribution

# In[59]:


# here we group labels and count the number of diseases type
label_count = Train_Classification.groupby('label', as_index=False).count()
label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
label_count['Label'] = label_count['Label'].apply(lambda x: disease_names[x])

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
font1 = {'family': 'Times New Roman','weight': 'bold','style':'normal','size': 20}
ax.set_xlabel('Type of disease',font1)
ax.set_ylabel('Count',font1)
ax = sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis')
ax.tick_params(labelsize=16)

plt.show()


# ## Images sampling

# In[60]:


# Sampling some images of CMD in the dataset 
def show_image(image_ids, labels):
    plt.figure(figsize=(15,10))
    
    for i, (image_id,label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3,3,i+1)
        img = cv2.imread(os.path.join('./Cassava leaf/train_images', image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(f"Class: {label}", fontsize=12)
        plt.axis("off")
    
    plt.show()
    
    
samples = Train_Classification.sample(9, random_state = 9417)
image_ids = samples['image_id'].values
labels = samples['Classification'].values

show_image(image_ids, labels)


# In[61]:


# Loading images function 
def load_img(path):
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[:, :, ::-1]
    return img_rgb


if True:
    img_shape = set()
    img_ext = set()
    img_names = Path('./Cassava leaf/train_images').glob('*')
    pbar = tqdm(img_names, total=len(Train_Classification))   
    # show the picture shapes and extensions
    for img_name in pbar:
        img = load_img(img_name.as_posix())
        img_shape.add(img.shape)
        img_ext.add(img_name.suffix)
    print(f'Image shapes are {img_shape}.')
    print(f'Image extensions are {img_ext}.')
else:
    print('Can not show the details of images')


# ## Images pixels analysis

# In[62]:


# explore the pixels of cassava leaf pictures
if True:
    img_names = Path('./Cassava leaf/train_images').glob('*')
    plt.figure(figsize=(10,10))
    pbar = tqdm(img_names, total=len(Train_Classification))
    for img_name in pbar:
        img = load_img(img_name.as_posix())
        # here use the calchist method in cv2 to show the histogram of photo pixels
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        plt.plot(hist)
    plt.show()
else:
    print('Can not show the details of images')


# ## Analyse Healthy and CMD Cassava Leaf(Pixel Density Distribution and RGB Channel Distribution)

# According to many studies, CMD is the most serious disease among all the Cassava leaf diseases, so here we compare the CMD to the healthy leaves

# In[63]:


Healthy_Cassava = Train_Classification[Train_Classification['Classification'] == 'Healthy']['image_id'].to_list()

CMD_Cassava = Train_Classification[Train_Classification['Classification'] == 'Cassava Mosaic Disease (CMD)']['image_id'].to_list()


# ### Healthy Cassava Leaves

# In[64]:


np.random.seed(20239417)

base_path = Path('./Cassava leaf')
train_img_dir =  base_path/'train_images'

random_images=[]
plt.figure(figsize=(16,12))
for i in range(9):
    random_images.append(np.random.choice(Healthy_Cassava))

for i in range(9):
    
    plt.subplot(3, 3, i + 1)
    img = plt.imread(train_img_dir/random_images[i])
    plt.imshow(img)
plt.show()   


# In[69]:


f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

font2 = {'family': 'Times New Roman','weight': 'bold','style':'normal','size': 14}
raw_image = plt.imread(train_img_dir/Healthy_Cassava[2])
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('Healthy Image',font2)
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)

plt.hist(raw_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
plt.hist(raw_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
plt.hist(raw_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
plt.xlabel('Intensity Value',font2)
plt.ylabel('Count',font2)
plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()


# In[70]:


imageArray=[]
for i in range(len(Healthy_Cassava)):
    imageArray.append(cv2.cvtColor(cv2.imread(str(train_img_dir/Healthy_Cassava[i])), cv2.COLOR_BGR2RGB))

red_values = []
green_values = []
blue_values = []
values = []

for i in range(len(imageArray)):
    red_values.append(np.mean(imageArray[i][:, :, 0]))
    green_values.append(np.mean(imageArray[i][:, :, 1]))
    blue_values.append(np.mean(imageArray[i][:, :, 2]))
    values.append(np.mean(imageArray[i]))

hist_data = [red_values, green_values, blue_values, values]
group_labels = ['Red', 'Green', 'Blue', 'All']

fig = ff.create_distplot(hist_data, group_labels,colors = ['red', 'green','blue','grey'])
fig.update_layout(template = 'plotly_white', title_text = 'Channel Distribution - Healthy')
fig.show()


# In[71]:


figData = []
for i, name in zip(range(3), ['Red', 'Green', 'Blue']):
    trace = go.Box(y = hist_data[i], name = name, boxpoints='all', marker_color  = name)
    figData.append(trace)

fig = go.Figure(figData)
fig.update_layout(title_text = 'Pixel Intensity Distribution - health leaf', template = 'plotly_white')
fig.show() 


# ### CMD Cassava Leaves

# In[72]:


np.random.seed(20239417)

random_images=[]
plt.figure(figsize=(16,12))
for i in range(9):
    random_images.append(np.random.choice(CMD_Cassava))

for i in range(9):
    
    plt.subplot(3, 3, i + 1)
    img = plt.imread(train_img_dir/random_images[i])
    plt.imshow(img)
plt.show()   


# In[73]:


f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)

raw_image = plt.imread(train_img_dir/CMD_Cassava[2])
plt.imshow(raw_image, cmap='gray')
plt.colorbar()
plt.title('CMD Image',font2)
print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
print(f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
print(f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

f.add_subplot(1,2, 2)

plt.hist(raw_image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)
plt.hist(raw_image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
plt.hist(raw_image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)
plt.xlabel('Intensity Value',font2)
plt.ylabel('Count',font2)
plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()


# In[74]:


imageArray=[]
for i in range(len(CMD_Cassava[:2000])):
    imageArray.append(cv2.cvtColor(cv2.imread(str(train_img_dir/CMD_Cassava[i])), cv2.COLOR_BGR2RGB))

red_values = []
green_values = []
blue_values = []
values = []

for i in range(len(imageArray)):
    red_values.append(np.mean(imageArray[i][:, :, 0]))
    green_values.append(np.mean(imageArray[i][:, :, 1]))
    blue_values.append(np.mean(imageArray[i][:, :, 2]))
    values.append(np.mean(imageArray[i]))

hist_data = [red_values, green_values, blue_values, values]
group_labels = ['Red', 'Green', 'Blue', 'All']

fig = ff.create_distplot(hist_data, group_labels,colors = ['red', 'green','blue','grey'])
fig.update_layout(template = 'plotly_white', title_text = 'Channel Distribution - CMD Images')
fig.show()


# In[75]:


figData = []
for i, name in zip(range(3), ['Red', 'Green', 'Blue']):
    trace = go.Box(y = hist_data[i], name = name, boxpoints='all', marker_color  = name)
    figData.append(trace)

fig = go.Figure(figData)
fig.update_layout(title_text = 'Pixel Intensity Distribution - CMD Images', template = 'plotly_white')
fig.show() 


# In[76]:


diseaseMapping = pd.read_json(base_path/'label_num_to_disease_map.json', typ='series')


# In[77]:


mappingDict = diseaseMapping.to_dict()


# In[78]:


Train_Classification = Train_Classification.replace(mappingDict)


# In[79]:


labelCounts = Train_Classification['label'].value_counts().reset_index()
labelCounts.columns = ['Label', 'Number of Observations']

colors = ['#9370DB','#00008B','#008B8B', '#2E8B57', '#FFD700']
fig = px.pie(labelCounts, 
             names = 'Label',values='Number of Observations', 
             labels = mappingDict, 
             title = 'Distribution of Labels in training set ',
             
             color_discrete_sequence=colors)
fig.update_layout(autosize=False, width=600, height=600)
fig.show()


# # END
