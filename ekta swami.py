#!/usr/bin/env python
# coding: utf-8

# In[27]:


from keras.applications import EfficientNetB0
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import Dropout, Input, Softmax, RandomFlip
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.utils import image_dataset_from_directory

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


# In[29]:


img_rows = 200
img_cols = 200
channels = 3

def load_dataset(path):
    dataset = image_dataset_from_directory(directory = path, 
                                           label_mode = 'categorical', 
                                           color_mode = 'rgb', 
                                           shuffle = False, 
                                           batch_size = None,
                                           image_size = (img_rows, img_cols),
                                           crop_to_aspect_ratio = True)
    return dataset

dataset1 = load_dataset('C:/Users/HP/Downloads/Brain-Tumor-Classification-DataSet-master/Brain-Tumor-Classification-DataSet-master/Training')
dataset2 = load_dataset('C:/Users/HP/Downloads/Brain-Tumor-Classification-DataSet-master/Brain-Tumor-Classification-DataSet-master/Testing')

images = []
labels = []
for (image, label) in dataset1:
    images.append(image)
    labels.append(label)
    
for (image, label) in dataset2:
    images.append(image)
    labels.append(label)
    
images = np.asarray(images)
labels = np.asarray(labels)
labels = labels.astype(np.uint8)

images, labels = shuffle(images, labels, random_state = 10)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 36)


# In[30]:


plt.figure(figsize = (50 * 3, 50))

for i in range(3):
    image = x_train[i].astype(np.uint8)
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)

plt.tight_layout()
plt.show()


# In[31]:


def create_model(dropout_rate = 0.5, flip_mode = None, lr = 0.001):
    model = Sequential()
    model.add(Input(shape = (img_rows, img_cols, channels)))
    if(flip_mode != None):
        model.add(RandomFlip(flip_mode))

    model.add(EfficientNetB0(weights = 'imagenet', include_top = False))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, activation = 'softmax'))

    model.build()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[32]:


reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, 
                              min_delta = 0.01, mode = 'auto', verbose = 1)


# In[33]:


model = create_model(dropout_rate = 0.5, flip_mode = 'horizontal')
history = model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test),
                    epochs = 2, batch_size = 32, callbacks = [reduce_lr])


# In[34]:


plt.rcParams.update({'font.size': 12})
def plot_performance(history, epochs, metric):
    fig, ax = plt.subplots(1, 1)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.title())
    
    x = list(range(1, epochs + 1))
    vy = history.history['val_' + metric]
    ty = history.history[metric]
    
    ax.plot(x, vy, 'b', label = 'Validation')
    ax.plot(x, ty, 'r', label = 'Train')
    
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[36]:


plot_performance(history, 2, 'accuracy')
plot_performance(history, 2, 'loss')


# In[ ]:




