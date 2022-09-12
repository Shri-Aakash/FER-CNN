# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:54:05 2022

@author: aakaa
"""

import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

cascadeClassifierPath=r'D:\Environments\ltts\Haar Cascade\haarcascade_frontalface_default.xml'
modelPath=r'D:\Deep Learning Codes\FER-CNN\FER200Epochs68%acc.h5'
faceFinder=cv2.CascadeClassifier(cascadeClassifierPath)
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
model=load_model(modelPath)
cap=cv2.VideoCapture(0)


while True:
    ret,img=cap.read()
    labels=[]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceFinder.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(img,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    
    