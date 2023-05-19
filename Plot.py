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

plt.rc('font',family='Times New Roman')

# you need summary.csv which record all matrices within the training. 
# summary.csv is provided in the same dir.
data = pd.read_csv('summary.csv')


# Plot of Precision and Recall(val)
precision_columns = [col for col in data.columns if col.endswith('v_precision')]
df = data[precision_columns]

colors = ['#63b2ee','#76da91','#f8cb7f', '#f89588', '#7cd6cf', '#7898e1','#9192ab', '#efa666', '#eddd86']

highest_precisions = df.max().sort_values()
sorted_precisions = highest_precisions.sort_values()

bar_width = 0.6
# Plot the bar chart with the specified colors and labels
fig, ax = plt.subplots(
                 facecolor='#E6E6E6',
                 edgecolor='black')
for i, model in enumerate(highest_precisions.index):
    ax.bar(i, highest_precisions[model], color=colors[i], label=model.split('_')[0],width=bar_width)
    ax.text(i, highest_precisions[model], f"{highest_precisions[model]:.4f}", ha='center', va='bottom')


ax.set_ylabel('Precision')
ax.set_title('Precision of Each Model')

# Show the legend and remove x-ticks
ax.legend(loc = 'lower right')
ax.set_xticks([])
plt.show()

recall_columns = [col for col in data.columns if col.endswith('v_recall')]
df = data[recall_columns]

colors = ['#63b2ee','#76da91','#f8cb7f', '#f89588', '#7cd6cf', '#7898e1','#9192ab', '#efa666', '#eddd86']

highest_recall = df.max().sort_values()

bar_width = 0.6
# Plot the bar chart with the specified colors and labels
fig, ax = plt.subplots(
                 facecolor='#E6E6E6',
                 edgecolor='black')
for i, model in enumerate(highest_recall.index):
    ax.bar(i, highest_recall[model], color=colors[i], label=model.split('_')[0],width=bar_width)
    ax.text(i, highest_recall[model], f"{highest_recall[model]:.4f}", ha='center', va='bottom')
ax.set_ylabel('Recall')
ax.set_title('Recall of Each Model')

# Show the legend and remove x-ticks
ax.legend(loc = 'lower right')
ax.set_xticks([])
plt.show()


# Validation Accuracy vs Epochs
acc_columns = [col for col in data.columns if col.endswith('v_accuracy')]
df = data[acc_columns]

fig, ax = plt.subplots(
                 facecolor='#E6E6E6',
                 edgecolor='black')

for index, model in enumerate(df.columns):
    if model == 'EfficientNetB7_v_accuracy':
        continue
    ax.plot(df.index, df[model], label=model.split('_')[0])

ax.plot(df.index, df['EfficientNetB7_v_accuracy'], label='EfficientNetB7', linewidth=2,linestyle='--',c = '#9192ab')

ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy %')
ax.set_title('Models Validation Accuracy vs. Epoch')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.xaxis.set_major_locator(ticker.MultipleLocator(1.000))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))


ax.set_ylim(0.4, 0.9)

ax.legend()
ax.grid(linestyle="--", linewidth=0.5, 
        color=".25", zorder=-10)
plt.show()


# Accuracy vs Epochs
acc_columns = [col for col in data.columns if col.split('_')[1]=='accuracy']
df = data[acc_columns]

fig, ax = plt.subplots(
                 facecolor='#E6E6E6',
                 edgecolor='black')

for model in df.columns:
    if model == 'EfficientNetB7_accuracy':
        continue
    ax.plot(df.index, df[model], label=model.split('_')[0])
ax.plot(df.index, df['EfficientNetB7_accuracy'], label='EfficientNetB7', linewidth=2,linestyle='--',c='#9192ab')

ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy %')
ax.set_title('Models Training Accuracy vs. Epoch')
ax.legend(loc='lower right')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.xaxis.set_major_locator(ticker.MultipleLocator(1.000))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))


ax.set_ylim(0.6, 0.9)
ax.grid(linestyle="--", linewidth=0.5, 
        color=".25", zorder=-10)
plt.show()

# loss vs. epoch
loss_columns = [col for col in data.columns if col.split('_')[1]=='loss']
df = data[loss_columns]

# Plot the dataframe
fig, ax = plt.subplots(
                 facecolor='#E6E6E6',
                 edgecolor='black')

for model in df.columns:
    if model == 'EfficientNetB7_loss':
        continue
    ax.plot(df.index, df[model], label=model.split('_')[0])
ax.plot(df.index, df['EfficientNetB7_loss'], label='EfficientNetB7', linewidth=2,linestyle='--',c='#9192ab')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Models Training Loss vs. Epoch')
ax.legend(loc='upper right')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.xaxis.set_major_locator(ticker.MultipleLocator(1.000))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))


ax.set_ylim(0.2,1.4)
ax.grid(linestyle="--", linewidth=0.5, 
        color=".25", zorder=-10)
plt.show()