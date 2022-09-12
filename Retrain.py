# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:47:16 2022

@author: aakaa
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

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

numTrainImg=0
for root,dirs,files in os.walk(dataDir):
    numTrainImg+=len(files)
    
numTestImg=0
for root,dirs,files in os.walk(testData):
    numTestImg+=len(files)
    
numEpochs=100

model=load_model()