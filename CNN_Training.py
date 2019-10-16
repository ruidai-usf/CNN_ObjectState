#!/usr/bin/env python

# coding: utf-8
# In[0]:
import os 
import shutil
import random

def Seperate_Train_Test_Data(sourceDir,  target_train_Dir, target_test_Dir, train_size):
    if not os.path.exists(target_train_Dir): 
        os.makedirs(target_train_Dir)
    else:
        for file in os.listdir(target_train_Dir): 
            targetFile = os.path.join(target_train_Dir,  file) 
            if os.path.isfile(targetFile): 
                os.remove(targetFile)     
    
    if not os.path.exists(target_test_Dir): 
        os.makedirs(target_test_Dir)
    else:
        for file in os.listdir(target_test_Dir): 
            targetFile = os.path.join(target_test_Dir,  file) 
            if os.path.isfile(targetFile): 
                os.remove(targetFile) 
    
    Control_num = 0
    file_list = os.listdir(sourceDir)
    train_num = int(train_size*len(file_list))
    random.shuffle(file_list)
    for file in file_list: 
        sourceFile = os.path.join(sourceDir,  file) 
        if Control_num<train_num:
            targetFile = os.path.join(target_train_Dir,  file) 
            if os.path.isfile(sourceFile): 
                shutil.copyfile(sourceFile, targetFile)
        else:
            targetFile = os.path.join(target_test_Dir,  file)
            if os.path.isfile(sourceFile): 
                shutil.copyfile(sourceFile, targetFile)
        Control_num+=1

#train_data_size = 0.8
#Seperate_Train_Test_Data("Origin_Data/creamy_paste", "train/creamy_paste", "test/creamy_paste", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/diced", "train/diced", "test/diced", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/grated", "train/grated", "test/grated", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/juiced", "train/juiced", "test/juiced", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/jullienne", "train/jullienne", "test/jullienne", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/sliced", "train/sliced", "test/sliced", train_data_size)
#Seperate_Train_Test_Data("Origin_Data/whole", "train/whole", "test/whole", train_data_size)

# In[1]:

import numpy as np
import gc

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def Load_Train_Test_Data(train_Dir, setting_data_size):
    files= os.listdir(train_Dir)
    data_length = int(setting_data_size*len(files))
    dataset_x_temp = np.zeros([data_length, 228, 228, 3], dtype = np.float16)
    temp_i = 0
    for file in files:
        if temp_i > data_length-1:
            break
        img_path = train_Dir+"/"+file;
        img = image.load_img(img_path, target_size=(228, 228))
        temp_x = image.img_to_array(img)
        temp_x = temp_x.reshape((1,) + temp_x.shape)
        dataset_x_temp[temp_i,:,:,:] = temp_x
        temp_i = temp_i+1
    return dataset_x_temp
    
Using_data_size = 1

##---------------------load train data-------------------------
# load and preprocess the dataset of images (creamy_paste)
train_x = Load_Train_Test_Data("train/creamy_paste", Using_data_size)

train_y1 = np.zeros([train_x.shape[0],7])
for i in range(train_y1.shape[0]):
    train_y1[i,0] = 1    

#print(train_x[:,0,0,0],'---------------',train_y1)

# load and preprocess the dataset of images (diced)
train_x_temp = Load_Train_Test_Data("train/diced", Using_data_size)

train_y2 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y2.shape[0]):
    train_y2[i,1] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)
    
#print(train_x[:,0,0,0],'---------------',train_y2)

# load and preprocess the dataset of images (grated)
train_x_temp = Load_Train_Test_Data("train/grated", Using_data_size)

train_y3 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y3.shape[0]):
    train_y3[i,2] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)

#print(train_x[:,0,0,0],'---------------',train_y3)

# load and preprocess the dataset of images (juiced)
train_x_temp = Load_Train_Test_Data("train/juiced", Using_data_size)

train_y4 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y4.shape[0]):
    train_y4[i,3] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)

#print(train_x[:,0,0,0],'---------------',train_y4)

# load and preprocess the dataset of images (jullienne)
train_x_temp = Load_Train_Test_Data("train/jullienne", Using_data_size)

train_y5 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y5.shape[0]):
    train_y5[i,4] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)

#print(train_x[:,0,0,0],'---------------',train_y5)

# load and preprocess the dataset of images (sliced)
train_x_temp = Load_Train_Test_Data("train/sliced", Using_data_size)

train_y6 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y6.shape[0]):
    train_y6[i,5] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)

#print(train_x[:,0,0,0],'---------------',train_y6)

# load and preprocess the dataset of images (whole)
train_x_temp = Load_Train_Test_Data("train/whole", Using_data_size)

train_y7 = np.zeros([train_x_temp.shape[0],7])
for i in range(train_y7.shape[0]):
    train_y7[i,6] = 1  

train_x = np.concatenate((train_x,train_x_temp),axis=0)

#print(train_x[:,0,0,0],'---------------',train_y7)

train_y = np.concatenate((train_y1,train_y2,train_y3,train_y4,
                            train_y5,train_y6,train_y7),axis=0)

del train_x_temp,train_y1,train_y2,train_y3,train_y4,train_y5,train_y6,train_y7
gc.collect()

#print(train_x[:,114,41,2],train_y)
# In[2]:

##---------------------load test data-------------------------
# load and preprocess the dataset of images (creamy_paste)
test_x = Load_Train_Test_Data("test/creamy_paste", Using_data_size)

test_y1 = np.zeros([test_x.shape[0],7])
for i in range(test_y1.shape[0]):
    test_y1[i,0] = 1    

# load and preprocess the dataset of images (diced)
test_x_temp = Load_Train_Test_Data("test/diced", Using_data_size)

test_y2 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y2.shape[0]):
    test_y2[i,1] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)
    
# load and preprocess the dataset of images (grated)
test_x_temp = Load_Train_Test_Data("test/grated", Using_data_size)

test_y3 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y3.shape[0]):
    test_y3[i,2] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)

# load and preprocess the dataset of images (juiced)
test_x_temp = Load_Train_Test_Data("test/juiced", Using_data_size)

test_y4 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y4.shape[0]):
    test_y4[i,3] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)

# load and preprocess the dataset of images (jullienne)
test_x_temp = Load_Train_Test_Data("test/jullienne", Using_data_size)

test_y5 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y5.shape[0]):
    test_y5[i,4] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)

# load and preprocess the dataset of images (sliced)
test_x_temp = Load_Train_Test_Data("test/sliced", Using_data_size)

test_y6 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y6.shape[0]):
    test_y6[i,5] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)

# load and preprocess the dataset of images (whole)
test_x_temp = Load_Train_Test_Data("test/whole", Using_data_size)

test_y7 = np.zeros([test_x_temp.shape[0],7])
for i in range(test_y7.shape[0]):
    test_y7[i,6] = 1  

test_x = np.concatenate((test_x,test_x_temp),axis=0)

test_y = np.concatenate((test_y1,test_y2,test_y3,test_y4,
                            test_y5,test_y6,test_y7),axis=0)

del test_x_temp,test_y1,test_y2,test_y3,test_y4,test_y5,test_y6,test_y7
gc.collect()

#print(test_x.shape,test_y.shape)

# In[3]:

# data random shuffle
ran_index_train = np.arange(train_x.shape[0])
ran_index_test = np.arange(test_x.shape[0])

np.random.shuffle(ran_index_train)
np.random.shuffle(ran_index_train)

np.random.shuffle(ran_index_test)
np.random.shuffle(ran_index_test)

temp_train_x = train_x
temp_train_y = train_y

temp_test_x = test_x
temp_test_y = test_y

train_x = temp_train_x[ran_index_train,:,:,:]
train_y = temp_train_y[ran_index_train,:]
test_x = temp_test_x[ran_index_test,:,:,:]
test_y = temp_test_y[ran_index_test,:]

#print(train_x[:,114,41,2],train_y)

del ran_index_train,ran_index_test,temp_train_x,temp_train_y,temp_test_x,temp_test_y
gc.collect()

#Preprocessing data
train_x = train_x.astype('float16')
test_x = test_x.astype('float16')

#centralization
#train_x = train_x - np.mean(train_x, axis = 0)
#test_x = test_x - np.mean(dataset_x, axis = 0)
#normalization
train_x = train_x/255
test_x = test_x/255

#print(train_x.shape,train_y.shape)
#print(test_x.shape,test_y.shape)
#
#print(train_y, test_y)

# In[4]:


import keras
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, Dropout, Activation, Flatten, MaxPooling2D

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = Convolution2D(32, (5,5))(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
#x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
# and a output layer 
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all convolutional VGG16 layers
for layer in base_model.layers:
    layer.trainable = True

# compile the model
model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), 
              loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()


# In[5]:


#datagen = ImageDataGenerator(
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

#datagen.fit(train_x)

#steps_long = int(0.5*len(train_x))

# train the model
fit_record = model.fit(train_x, train_y, validation_split=0.2, verbose=1, epochs=10)
#model.fit_generator(datagen.flow(train_x, train_y, batch_size=5),steps_per_epoch=steps_long, epochs=20)
model.save_weights('weights_ft4.h5')
model.save('model_ft4.h5')

train_loss=fit_record.history['loss']
valid_loss=fit_record.history['val_loss']
train_acc=fit_record.history['acc']
valid_acc=fit_record.history['val_acc']

np.save('train_loss_ft4', train_loss)
np.save('valid_loss_ft4', valid_loss)
np.save('train_acc_ft4', train_acc)
np.save('valid_acc_ft4', valid_acc)
# In[6]:

# test the model
scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

