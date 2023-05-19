from tensorflow import keras
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import skimage.io as io
from skimage import data_dir
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import data_dir,io,color
import h5py
from keras import Model
from keras.models import load_model
from sklearn.manifold import TSNE

# IMPORTANT
# replace your dir after downloading the Cassava dataset from Kaggle.
# Download Link: https://drive.google.com/drive/folders/1RqifyoVkT242wa4k5BqbHHjZmSvqlLr6?usp=share_link
work_dir = 'E:/cassava-leaf-disease-classification/'
train_path = 'E:/cassava-leaf-disease-classification/train_images'

# you also need .h5 
# Download Link: https://drive.google.com/drive/folders/13JdjrfLRhFBBaJgzSzIqY5IexxNdv1rP?usp=share_link
model = load_model('Cassava_Leaf_Disease_Detection_EfficientNetB7_model_BestModel.h5')

# data loading
data = pd.read_csv(work_dir + 'train.csv')

# Load the JSON file containing labels
with open(work_dir + 'label_num_to_disease_map.json') as json_file:
    label_mapping = json.load(json_file)
    label_mapping = {int(key): value for key, value in label_mapping.items()}

# Configure the working dataset
data['class_name'] = data['label'].apply(lambda x: label_mapping[x])

def remove_jpg(x):
    return int(x.split(".jpg")[0])

data['image_id'] = data['image_id'].apply(lambda x: remove_jpg(x))

data.sort_values(by=['image_id'],axis=0,ascending=True,inplace=True)
data = data.reset_index(drop=True)

label = data['label']

def convert_gray(f):
    rgb=Image.open(f)
    x = rgb.resize((224,224))
    return np.asarray(x)
coll = io.ImageCollection(train_path+'/*.jpg',load_func=convert_gray)
mat=io.concatenate_images(coll)

def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path) 
    try: 
        for layer, g in f.items():  
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): 
                print("      {}: {}".format(key, value))            
    finally:
        f.close()
 

print_keras_wegiths('Cassava_Leaf_Disease_Detection_EfficientNetB7_model_BestModel.h5')

print("Using loaded model to predict...")
load_model = load_model('Cassava_Leaf_Disease_Detection_EfficientNetB7_model_BestModel.h5')
model = Model(inputs=load_model.input,outputs=load_model.get_layer('global_average_pooling2d_1').output)

# change this to 10000! since it spend long time to transfer, only set sample length to be 1000 as demo.
sample_len = 1000 #[0,21397]

tsne_data = mat[:sample_len]

predict = model.predict(tsne_data)
tsne = TSNE(n_components=2,learning_rate=sample_len,init = 'pca',random_state=0)
X_tsne_0 = tsne.fit_transform(predict)

#X_tsne_0 = X_tsne_0.reshape((sample_len, X_tsne_0.shape[1]))
y = tsne.fit_transform(X_tsne_0)


color = np.stack(label, axis=0)
color_sample = color[:sample_len]


ts = TSNE(n_components=2, learning_rate=sample_len, init='pca', random_state=0)

tsne_data = tsne_data.reshape((len(tsne_data), 224*224*3))
y_original = ts.fit_transform(tsne_data)


# Function to change the value
def change_value(value):
    if value == 'Healthy':
        return value
    return value.split('(')[1].replace(')','')

# Update all values in the dictionary using dictionary comprehension
new_mapping = {key: change_value(value) for key, value in label_mapping.items()}

color_list = ['#63b2ee','#76da91','#f8cb7f', '#f89588', '#7cd6cf', '#9192ab', '#7898e1', '#efa666', '#eddd86']
plt.rc('font',family='Times New Roman')

fig = plt.figure(figsize=(8, 16),facecolor='#E6E6E6',edgecolor='black')
ax1 = fig.add_subplot(3, 1, 3)
# color_list = ['#8B0000', '#FF6A6A', '#00FF00', '#F4A460', '#00CDCD', '#0000FF']
for i in range(sample_len):
    if color[i] == 0:
        s0 = plt.scatter(y_original[i, 0], y_original[i, 1], c=color_list[0], s=8)
    elif color[i] == 1:
        s1 = plt.scatter(y_original[i, 0], y_original[i, 1], c=color_list[1], s=6)
    elif color[i] == 2:
        s2 = plt.scatter(y_original[i, 0], y_original[i, 1], c=color_list[2], s=6)
    elif color[i] == 3:
        s3 = plt.scatter(y_original[i, 0], y_original[i, 1], c=color_list[3], s=6)
    elif color[i] == 4:
        s4 = plt.scatter(y_original[i, 0], y_original[i, 1], c=color_list[4], s=6)

plt.legend((s0,s1,s2,s3,s4),(new_mapping[0],new_mapping[1],new_mapping[2]
,new_mapping[3],new_mapping[4]) ,loc = 'best',markerscale = 2)

ax1.set_title('Dataset dimensionality reduction using t-SNE', fontsize=14)
plt.show()

fig = plt.figure(figsize=(8, 16),facecolor='#E6E6E6',edgecolor='black')
ax1 = fig.add_subplot(3, 1, 3)
# color_list = ['#8B0000', '#FF6A6A', '#00FF00', '#F4A460', '#00CDCD', '#0000FF']
for i in range(sample_len):
    if color[i] == 0:
        s0 = plt.scatter(y[i, 0], y[i, 1], c=color_list[0], s=6)
    elif color[i] == 1:
        s1 = plt.scatter(y[i, 0], y[i, 1], c=color_list[1], s=6)
    elif color[i] == 2:
        s2 = plt.scatter(y[i, 0], y[i, 1], c=color_list[2], s=6)
    elif color[i] == 3:
        s3 = plt.scatter(y[i, 0], y[i, 1], c=color_list[3], s=6)
    elif color[i] == 4:
        s4 = plt.scatter(y[i, 0], y[i, 1], c=color_list[4], s=6)

plt.legend((s0,s1,s2,s3,s4),(new_mapping[0],new_mapping[1],new_mapping[2]
,new_mapping[3],new_mapping[4]) ,loc = 'best',markerscale = 2)

ax1.set_title('Output features dimensionality reduction using t-SNE', fontsize=14)

plt.show()