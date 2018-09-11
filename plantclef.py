from keras.models import Sequential
from scipy.misc import imread

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense,Input
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from scipy.misc import imresize
import keras.applications
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.layers import Dropout
import np_utils


import os
import glob
path = os.getcwd()+ "/akash/data/"
path2 = []
for dir in os.listdir(path):
    path2.append(path  +dir)

train = []
for path in path2:
    cl= path.split("/")[5]
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            #filename = file.split(".")[0]
               
            train.append(list([cl , path +"/"+ file]))
            
        
import random
random.shuffle(train)
count = []
for i in train:
    count.append(i[0])
import collections
x = collections.Counter(count) 
import operator
sorted_x = sorted(x.items(), key=operator.itemgetter(1))

fil = [x[0] for x in sorted_x[-10:]]


x_data = []
y_data = []
for im in train:
    if im[0] in fil:
        x_data.append(im[1])
        y_data.append(im[0])

x_train = x_data[:6970]
x_test = x_data[6970:]
y_train = y_data[:6970]
y_test = y_data[6970:]


from numpy import array
from keras.utils import to_categorical
num_classes = 10
y_train = array(y_train)
y_test = array(y_test)

def onehot(labels,custom_n_uniques=None):
    from keras.utils.np_utils import to_categorical
    uniques, ids = np.unique(labels,return_inverse=True)
    if custom_n_uniques is None:
        return to_categorical(ids,len(uniques))
    else:
        return to_categorical(ids,custom_n_uniques)
y_train = onehot(y_train, num_classes)
y_test = onehot(y_test, num_classes)


def generator(features, labels, batch_size):
    
    while(1):
        j = 0
        while(j+batch_size-1 < len(features)):
            x_batch = []
            y_batch = []
            for i in range(batch_size):
                    #index= random.choice(len(features),1)
                x =image.load_img(features[i+j],target_size=(224,224))
                image1 = image.img_to_array(x)
                    # reshape data for the model
                img = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
                x = preprocess_input(img)[0]
                y = labels[i]
                x_batch.append(x)
                y_batch.append(y)


            yield (np.array(x_batch),np.array(y_batch))

            j += batch_size
            

model = keras.applications.VGG16(weights='imagenet', include_top=False)
input = Input(shape=(224,224,3),name = 'image_input')
model2 = model(input)
for layer in model.layers: layer.trainable = False
x = Flatten(name='flatten')(model2)
y = Dense(20, activation='relu', name='dense1')(x)
z = Dropout(0.2)(y)
last = Dense(10, activation='softmax', name='dense2')(z)
new_model = Model(inputs=input, outputs=last)


batch_size = 32
gen = generator(x_train , y_train , 32)
new_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

total = len(y_train)
new_model.fit_generator(gen, samples_per_epoch=total/batch_size,nb_epoch=5)

import h5py
model.save('my_model.h5')

