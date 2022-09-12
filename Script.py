# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:11:11 2022

@author: aakaa
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPooling2D
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

ImgH=48
ImgW=48
bSize=16
dataDir=r'D:\Datasets\FER\masmbre\archive (1)\train'
categories=os.listdir(dataDir)

testData=r'D:\Datasets\FER\masmbre\archive (1)\test'
trainDataGen=ImageDataGenerator(rescale=1./255,
                                rotation_range=30,
                                shear_range=0.3,
                                zoom_range=0.3,
                                horizontal_flip=True,
                                fill_mode='nearest')

validationDataGen=ImageDataGenerator(rescale=1./255)

trainData=trainDataGen.flow_from_directory(directory=dataDir,
                                           target_size=(ImgH,ImgW),
                                           color_mode='grayscale',
                                           class_mode='categorical',
                                           batch_size=bSize,
                                           shuffle=True)

validationData=validationDataGen.flow_from_directory(directory=testData,
                                                     target_size=(ImgH,ImgW),
                                                     color_mode='grayscale',
                                                     class_mode='categorical',
                                                     batch_size=bSize,
                                                     shuffle=True)

img,label=trainData.__next__()

i=random.randint(0,(img.shape[0])-1)
image=img[i]
labl=categories[label[i].argmax()]
cv2.imshow(labl,image)
cv2.waitKey(0)
cv2.destroyAllWindows()


model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,1)))
model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.15))

model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.15))

model.add(Conv2D(256, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.15))

model.add(Conv2D(512, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


numTrainImg=0
for root,dirs,files in os.walk(dataDir):
    numTrainImg+=len(files)
    
numTestImg=0
for root,dirs,files in os.walk(testData):
    numTestImg+=len(files)
    
numEpochs=100
modelHistory=model.fit(trainData,
                       steps_per_epoch=numTrainImg//bSize,
                       epochs=numEpochs,
                       validation_data=validationData,
                       validation_steps=numTestImg//bSize)



model.save('FER100Epochs.h5')