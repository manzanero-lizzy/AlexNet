#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[33]:


path = 'C:\\Users\\Eliza Marie\\Anaconda\\train_folder'
#C:\Users\Eliza Marie\Anaconda\train_folder
train_datagen = ImageDataGenerator(rescale=1. / 255)
#change to categorical
train = train_datagen.flow_from_directory(path, target_size=(227,227), class_mode='categorical')


# In[35]:


type(train)


# In[36]:


type(train_datagen)


# In[37]:


print("Batch Size for Input Image : ",train[0][0].shape)
print("Batch Size for Output Image : ",train[0][1].shape)
print("Image Size of first image : ",train[0][0][0].shape)
print("Output of first image : ",train[0][1][0].shape)


# In[38]:


fig , axs = plt.subplots(2,3 ,figsize = (10,10))
axs[0][0].imshow(train[0][0][12])
axs[0][0].set_title(train[0][1][12])
axs[0][1].imshow(train[0][0][10])
axs[0][1].set_title(train[0][1][10])
axs[0][2].imshow(train[0][0][5])
axs[0][2].set_title(train[0][1][5])
axs[1][0].imshow(train[0][0][20])
axs[1][0].set_title(train[0][1][20])
axs[1][1].imshow(train[0][0][25])
axs[1][1].set_title(train[0][1][25])
axs[1][2].imshow(train[0][0][3])
axs[1][2].set_title(train[0][1][3])


# In[45]:


def AlexNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(4096, activation = 'relu', name = "fc0")(X)
    
    X = Dense(4096, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(2,activation='softmax',name = 'fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AlexNet')
    return model


# In[46]:


alex = AlexNet(train[0][0].shape[1:])


# In[47]:


alex.summary()


# In[48]:


alex.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[75]:


alex.fit_generator(train,epochs=5)


# In[76]:


path_test = 'C:\\Users\\Eliza Marie\\Anaconda\\val_folder'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(path_test, target_size=(227,227), class_mode='categorical')


# In[77]:


preds = alex.evaluate_generator(test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[85]:


path_test = 'C:\\Users\\Eliza Marie\\Anaconda\\test_folder'
predict_datagen = ImageDataGenerator(rescale=1. / 255)
predict = predict_datagen.flow_from_directory(path_test, target_size=(227,227), batch_size = 1,class_mode='categorical')


# In[86]:


predictions = alex.predict_generator(predict)


# In[106]:


imshow(predict[70][0][0])


# In[109]:


values = predictions[70]
#print(predictions[70])
floats = [float(x) for x in values]
print(floats)


# In[89]:


import os 
def get_category(predicted_output):
    path ="C:\\Users\\Eliza Marie\\Anaconda\\train_folder"
    return os.listdir(path)[np.argmax(predicted_output)]


# In[96]:


print(get_category(predictions[190]))


# In[91]:


fig , axs = plt.subplots(2,3 ,figsize = (10,10))

axs[0][0].imshow(predict[10][0][0])
axs[0][0].set_title(get_category(predictions[10]))

axs[0][1].imshow(predict[22][0][0])
axs[0][1].set_title(get_category(predictions[22]))

axs[0][2].imshow(predict[100][0][0])
axs[0][2].set_title(get_category(predictions[100]))

axs[1][0].imshow(predict[210][0][0])
axs[1][0].set_title(get_category(predictions[210]))

axs[1][1].imshow(predict[188][0][0])
axs[1][1].set_title(get_category(predictions[188]))

axs[1][2].imshow(predict[200][0][0])
axs[1][2].set_title(get_category(predictions[200]))


# In[83]:


imshow(predict[200][0][0])


# In[110]:


import visualkeras
visualkeras.layered_view(alex)


# In[ ]:




