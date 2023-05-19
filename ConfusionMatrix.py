import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
import json
from PIL import Image
import re
from skimage import data_dir,io,color

# IMPORTANT
# change your dir after downloading the Cassava dataset from Kaggle.
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

# Load the data and true labels, we only use 10000 sample in demo
data = mat[:10000]
true_labels = label[:10000]

# Ensure the true_labels are one-hot encoded
if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:
    true_labels = np.argmax(true_labels, axis=1)

# Predict the labels
predicted_labels = model.predict(data,verbose=1)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
# Define the class names
class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()