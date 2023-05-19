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


# # Modelling

# In[20]:


# checking if training uses the apple silcon 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_built_with_gpu_support()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[21]:


# Set the path of input images 
root_path = './Cassava leaf/'
training_img_path = root_path + 'train_images/'
test_img_path = root_path + 'test_images/'
model_path = './input/cassavanet-baseline-models/'


# In[22]:


# Set the input of image size and batch size during the modelling
# when using the efficientnetb7 model to train, here we make batch size equal 16, because this model is more complex, it takes up more memory
image_size = 224
batch_size = 16
# batch_size = 64


# Reread the training dataset
train_dataset = pd.read_csv('./Cassava leaf/train.csv')

# Transform the attribute "label" to the string
train_dataset["label"] = train_dataset["label"].astype(str)


# ## Data Augmentation

# In[23]:


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

# parameter adjustment：
# rotation_range=40(270), width_shift_range=0.2, height_shift_range=0.2(0.3), remove brightness_range=[0.1,0.9]
# remove channel_shift_range=0.1, remove rescale=1/255,shear_range=0.2,zoom_range=0.2(0.3)

Data_Generator = ImageDataGenerator(rotation_range=40, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
        
                                    shear_range = 0.2, 
                                    zoom_range = 0.2, 
                                 
                                    horizontal_flip = True, 
                                    vertical_flip = True, 
                           
                                    validation_split = 0.2,
                                    fill_mode = 'nearest')

# remove rescale=1/255
Valid_Generator = ImageDataGenerator(validation_split = 0.2)


tr_dataset = Data_Generator.flow_from_dataframe(dataframe=train_dataset, directory = root_path + "train_images", seed = 9417, 
x_col = "image_id", y_col = "label", interpolation = 'nearest', target_size = (image_size, image_size), class_mode = "categorical", 
batch_size = batch_size, shuffle = True, subset = "training")


valid_dataset = Valid_Generator.flow_from_dataframe(dataframe=train_dataset, directory = root_path + "train_images", seed = 9417, 
x_col = "image_id",y_col = "label", interpolation = 'nearest', target_size = (image_size, image_size), class_mode = "categorical",
batch_size = batch_size, shuffle = True, subset = "validation")


# ## EfficientNetB7

# In[24]:


# Define the loss function, according to many studies, we know that cross entropy is suitable for this topic
loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False,label_smoothing=0.0001,name='categorical_crossentropy')

# After setting base model, here add dense layer with elu activation function and dropout layer following
# elu activation is more steady than relu activation because relu activation can not deal with the negative values better
# dropout operation can help reduce part of parameters which makes the model not be over-fitted
# GlobalAveragePooling can can significantly reduce the number of parameters, enhance the accuracy and stability of the model, and suppress the phenomenon of overfitting in the network
# dense layer extract the previous features and do non-linear operations to make them mapped into the output space
# Because these models have large number of parameters, adding L1/L2 regularizer can limit the complexity of the model to achieve a balance between complexity and performance
# flatten layer is used to flatten the mutiple dimensional input

def EfficientNetB7_Model(weights = None):
    
    base_model = EfficientNetB7(include_top = False,weights = weights, input_shape = [image_size,image_size,3])
    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.6))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='elu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='elu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = 'softmax'))
    # here we use three evaluation indexes, accuracy, precision and recall
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.001), metrics = ["categorical_accuracy",keras.metrics.Precision(),keras.metrics.Recall()])
    return model


# In[25]:


# In order to obtain more accuracy in this problem, we pre-train the network using the imagenet 
EfficientNetB7_model = EfficientNetB7_Model(weights = "imagenet")
EfficientNetB7_model.summary()


# In[26]:


# Plot the neural network structure
plot_model(EfficientNetB7_model,to_file='EfficientNetB7_model.png',show_shapes=True)


# ## Model Training

# In[27]:


# The function of ReduceLROnPlateau is to reduce the learning rate when the monitored quantity no longer improves or decreases. 
# When learning stops, the model will always benefit from reducing the learning rate by 2-10 times.
# here we set the monitor variable is loss
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, mode='min', verbose=1)

#patience = 3
# The function of EarlyStopping is to stop the training to prevent overfitting when the monitored quantity no longer increases or decreases.
early_stopping = EarlyStopping(monitor = "val_loss", mode = "min", patience = 7, restore_best_weights=True, verbose=1)

# ModelCheckpoint is used to save the best model during the process of training 
check_point = tf.keras.callbacks.ModelCheckpoint("./Cassava_Leaf_Disease_Detection_EfficientNetB7_BestModel.h5", 
                                                 monitor = 'val_loss',mode = 'min', save_best_only=True)

# set epochs equal 20 to train
# here we test the result of training, 20 is a appropriate number, since after 20 epochs, the effect will be worse or nearly constant
history_EfficientNetB7 = EfficientNetB7_model.fit(tr_dataset, validation_data = valid_dataset, epochs = 20, 
                                      callbacks = [early_stopping, reduce_lr, check_point])


# In[26]:


# Models Evaluation 
accuracy = history_EfficientNetB7.history["categorical_accuracy"]
v_accuracy = history_EfficientNetB7.history["val_categorical_accuracy"]

loss = history_EfficientNetB7.history["loss"]
v_loss = history_EfficientNetB7.history["val_loss"]

precision = history_EfficientNetB7.history["precision"]
v_precision = history_EfficientNetB7.history["val_precision"]

recall = history_EfficientNetB7.history["recall"]
v_recall = history_EfficientNetB7.history["val_recall"]

# Plot four main Evaluation criterions
epochs = range(20)

plt.figure(figsize = (12, 12))
plt.subplot(2, 2, 1)
plt.plot(epochs, acc, label = "Training Set Accuracy")
plt.plot(epochs, v_acc, label = "Validation Set Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 2, 2)
plt.plot(epochs, loss, label = "Training Set Loss")
plt.plot(epochs, v_loss, label = "Validation Set Loss")
plt.legend(loc = "lower right")
plt.title("Training and Validation Loss")
plt.show()

plt.subplot(2, 2, 3)
plt.plot(epochs, precision, label = "Training Set Precision")
plt.plot(epochs, v_precision, label = "Validation Set Precision")
plt.legend(loc = "lower right")
plt.title("Training and Validation Precision")
plt.show()

plt.subplot(2, 2, 4)
plt.plot(epochs, recall, label = "Training Set Recall")
plt.plot(epochs, v_recall, label = "Validation Set Recall")
plt.legend(loc = "lower right")
plt.title("Training and Validation Recall")
plt.show()


# In[ ]:


accuracy


# In[ ]:


v_accuracy


# In[ ]:


loss


# In[ ]:


v_loss


# In[ ]:


precision


# In[ ]:


v_precision


# In[ ]:


recall


# In[ ]:


v_recall


# ## Model Prediction

# In[39]:


# load model
EfficientNetB7_model =  tf.keras.models.load_model("./Final Models/Cassava_Leaf_Disease_Detection_EfficientNetB7_model_BestModel.h5")


# In[27]:


# according to the requirement of competition, we also need to predict the test image and submit the result of prediction in the csv file
preds = []
sample_sub_csv = pd.read_csv('./Cassava leaf/sample_submission.csv')

for image in sample_sub_csv.image_id:
    img = tf.keras.preprocessing.image.load_img('./Cassava leaf/test_images/' + image)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (image_size, image_size))
    img = tf.reshape(img, (-1, image_size, image_size, 3))
    prediction = EfficientNetB7_model.predict(img/255)
    preds.append(np.argmax(prediction))

final_submission = pd.DataFrame({'image_id': sample_sub_csv.image_id, 'label': preds})
final_submission.to_csv('submission.csv', index=False) 


# In[28]:


print(final_submission.head())


# # END
