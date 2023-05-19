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


# ## EfficientNetB3

# ## Model Preprocessing

# In[ ]:


# Defining the working directories
work_dir = 'E:/cassava-leaf-disease-classification/'
train_path = 'E:/cassava-leaf-disease-classification/train_images'


# In[ ]:


# data loading
data = pd.read_csv(work_dir + 'train.csv')
print(data['label'].value_counts())


# In[ ]:


# Load the JSON file containing labels
with open(work_dir + 'label_num_to_disease_map.json') as json_file:
    label_mapping = json.load(json_file)
    label_mapping = {int(key): value for key, value in label_mapping.items()}

# Configure the working dataset
data['class_name'] = data['label'].apply(lambda x: label_mapping[x])

label_mapping


# In[ ]:


# generate train and test sets
train, test = train_test_split(data, test_size = 0.2, random_state = 9417, stratify = data['class_name'])


# In[ ]:


# Set the input format of pictures
IMG_SIZE = 224
size = (IMG_SIZE,IMG_SIZE)
n_CLASS = 5
# When training CNN model, we adjust batch size to 32
BATCH_SIZE = 15


# ##  Data Augmentation

# In[ ]:


# Parameters: rotation_range is the angle of rotating images when data augmentation 
# width_shift_range is the proportion of the iamge width, which means the range of image shifts horizontally when data augmentation
# height_shift_range is the proportion of the iamge height, which means the range of image shifts vertically when data augmentation
# horizontal_flip means conducting filp horizontally randomly
# vertical_flip means conducting filp vertically randomly
# shear_range means cutting degrees
# zoom_range means zooming degrees
# channel_shift_range means the degrees of randomly shifting
# rescale means the factor of rescaling
# fill_mode means the points which exceeds the boundary will be operated by given method
datagen_train = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest',
)

datagen_val = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
)

train_set = datagen_train.flow_from_dataframe(
    train,
    directory=train_path,
    seed=9417,
    x_col='image_id',
    y_col='class_name',
    target_size = size,
    class_mode='categorical',
    interpolation='nearest',
    shuffle = True,
    batch_size = BATCH_SIZE,
)

test_set = datagen_val.flow_from_dataframe(
    test,
    directory=train_path,
    seed=9417,
    x_col='image_id',
    y_col='class_name',
    target_size = size,
    class_mode='categorical',
    interpolation='nearest',
    shuffle=True,
    batch_size=BATCH_SIZE,    
)


# ## Model Training

# In[ ]:


def create_model():
    
    model = Sequential()
    # initialize the model with input shape
    model.add(
        EfficientNetB3(
            input_shape = (IMG_SIZE, IMG_SIZE, 3), 
            include_top = False,
            weights='imagenet',
            drop_connect_rate=0.6,
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(
        256, 
        activation='relu', 
        bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
    ))
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation = 'softmax'))
    
    return model

leaf_model = create_model()
leaf_model.summary()


# In[ ]:


keras.utils.plot_model(leaf_model,to_file='graph.png',show_shapes=True)


# In[ ]:


EPOCHS = 20
STEP_SIZE_TRAIN = train_set.n // train_set.batch_size
STEP_SIZE_TEST = test_set.n // test_set.batch_size
# starter_learning_rate = 0.1
#1e-3
# ITERS = 200     
# LR =tf.compat.v1.train.exponential_decay(starter_learning_rate,ITERS,100,0.8,staircase=True)


# In[ ]:


def model_fit():
    leaf_model = create_model()
    
    # Loss function 
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits = False,
        label_smoothing=0.0001,
        name='categorical_crossentropy'
    )
    
    # Compile the model
    leaf_model.compile(
        optimizer = Adam(learning_rate = 1e-3),
        # optimizer = Adam(learning_rate = LR),
        loss = loss, #'categorical_crossentropy'
        metrics = ['categorical_accuracy',keras.metrics.Precision(),keras.metrics.Recall()]
    )
    
    # Stop training when the val_loss has stopped decreasing for 3 epochs.
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=5,
        restore_best_weights=True, 
        verbose=1,
    )
    
    # Save the model with the minimum validation loss
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        "Cassava_best_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
    )
    
    # Reduce learning rate once learning stagnates
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        mode='min',
        verbose=1,
    )
    
    # Fit the model
    history = leaf_model.fit(
        train_set,
        validation_data=test_set,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_steps=STEP_SIZE_TEST,
        callbacks=[es, checkpoint_cb, reduce_lr],
    )
    
    # Save the model
    leaf_model.save('Cassava_model'+'.h5')  
    
    return history


# In[ ]:


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from tensorflow.compat.v1.keras import backend as K
K.set_session(sess)


# In[ ]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.compat.v1.Session(config=config)


# In[ ]:


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
try:
    final_model = keras.models.load_model('Cassava_model.h5')
except Exception as e:
    with tf.device('/GPU:0'):
        results = model_fit()
    print('Train Categorical Accuracy: ', max(results.history['categorical_accuracy']))
    print('Test Categorical Accuracy: ', max(results.history['val_categorical_accuracy']))


# In[ ]:


str(results.history)


# In[ ]:


# Write these evaluation values into the txt file
with open('results.txt','w') as f:
    for line in str(results.history):
        f.write(line)


# In[ ]:


def trai_test_plot(acc, test_acc, loss, test_loss):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize= (15,10))
    fig.suptitle("Model's metrics comparisson", fontsize=20)

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(test_acc) + 1), test_acc)
    ax1.set_title('History of Accuracy', fontsize=15)
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Accuracy', fontsize=15)
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(test_loss) + 1), test_loss)
    ax2.set_title('History of Loss', fontsize=15)
    ax2.set_xlabel('Epochs', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    ax2.legend(['training', 'validation'])
    plt.show()
    

trai_test_plot(
    results.history['categorical_accuracy'],
    results.history['val_categorical_accuracy'],
    results.history['loss'],
    results.history['val_loss']
)


# # END
