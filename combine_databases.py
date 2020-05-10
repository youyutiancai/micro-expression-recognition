#!/usr/bin/env python
# coding: utf-8

# In[4]:


from imutils import face_utils
import numpy as np
import imutils
import cv2
import dlib
import collections
import matplotlib.pyplot as plt
import pandas as pd
from os import path
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


shape_predictor = "shape_predictor_68_face_landmarks.dat"


# In[6]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


# In[16]:


def writeImage(part, imagePath, imageName, partName):
    image = imagePath + imageName + ".jpg"
    image = cv2.imread(image)
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = cv2.boundingRect(part)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if w < 0:
        w = 0
    if h < 0:
        h = 0
    roi = image[y:y + h, x:x + w]
#     roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    cv2.imwrite(imagePath + imageName + "_" + partName + ".jpg", roi)


# In[8]:


def segmentLandmarks(imagePath, imageName):  
    # load the input image, resize it, and convert it to grayscale
    image = imagePath + imageName + ".jpg"
    image = cv2.imread(image)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    faceparts = dict()
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            faceparts[name] = shape[i:j]
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
#             clone = image.copy()
#             cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 0, 255), 2)

#             # loop over the subset of facial landmarks, drawing the
#             # specific face part
#             for (x, y) in shape[i:j]:
#                 cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
#             # extract the ROI of the face region as a separate image
#             (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
#             if x < 0:
#                 x = 0
#             if y < 0:
#                 y = 0
#             if w < 0:
#                 w = 0
#             if h < 0:
#                 h = 0
#             roi = image[y:y + h, x:x + w]
# #             print(shape[i:j])
#             roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

# #             print landmarks
#             cv2.imshow("ROI", roi)
#             cv2.imshow("Image", clone)
#             cv2.waitKey(0)

#   store the landmarks
    rightEyeParts = np.concatenate((faceparts['right_eyebrow'], faceparts['right_eye']))
    writeImage(rightEyeParts, imagePath, imageName, 'right_eye')
    leftEyeParts = np.concatenate((faceparts['left_eyebrow'], faceparts['left_eye']))
    writeImage(leftEyeParts, imagePath, imageName, 'left_eye')
    writeImage(faceparts['mouth'], imagePath, imageName, 'mouth')
    return faceparts


# In[21]:


def loadAllFile(database, databaseName, i, j):
    fileName = ''
    imageName = ''
    if databaseName == 'CASME2':
        fileName = 'sub0' + str(database.Subject[i])
        if database.Subject[i] >= 10:
            fileName = 'sub' + str(database.Subject[i])
        fileName = fileName + '/' + database.Filename[i] + '/'
        imageName = 'reg_img'
    if databaseName == 'CASME':
        fileName = 'sub0' + str(database.Subject[i])
        if database.Subject[i] >= 10:
            fileName = 'sub' + str(database.Subject[i])
        fileName = fileName + '/' + database.Filename[i] + '/'
        imageName = 'reg_' + database.Filename[i] + '-' 
        if not(path.exists('./database/' + databaseName + '/Cropped/' + fileName + imageName + str(int(database.OnsetFrame[i]) + j) + '.jpg')):
            foreName = '-0' 
            if int(database.OnsetFrame[i]) + j < 10:
                foreName = '-00' 
            imageName = 'reg_' + database.Filename[i] + foreName
    if databaseName == 'SAMM':
        foreName = '00' + str(database.Subject[i])
        if database.Subject[i] >= 10:
            foreName = '0' + str(database.Subject[i])
        fileName = foreName + '/' + database.Filename[i] + '/'
        imageName = foreName + '_' 
#                 print('./database/' + databaseName + '/Cropped/' + fileName + imageName + '.jpg')
        if not(path.exists('./database/' + databaseName + '/Cropped/' + fileName + imageName + str(int(database.OnsetFrame[i]) + j) + '.jpg')):
            imageName = '0' 
            if int(database.OnsetFrame[i]) + j < 1000:
                imageName = '00' 
            imageName = foreName + '_' + imageName
        if not(path.exists('./database/' + databaseName + '/Cropped/' + fileName + imageName + str(int(database.OnsetFrame[i]) + j) + '.jpg')):
            imageName = '0' 
            if int(database.OnsetFrame[i]) + j < 100:
                imageName = '00'
            imageName = foreName + '_' + imageName
    imagePath = './database/' + databaseName + '/Cropped/' + fileName
    return imagePath, imageName


# In[22]:


def segmentAll(database, databaseName, emotions):
    for i in range(len(database)):
        emotion = database['Emotion'][i]
        if not(emotion in emotions):
            continue
        if database.ApexFrame[i] == "/":
            continue
        for j  in range(int(database.ApexFrame[i]) - int(database.OnsetFrame[i]) + 1):
            imagePath, imageName = loadAllFile(database, databaseName, i, j)
#             print(imagePath, imageName)
            segmentLandmarks(imagePath, imageName + str(int(database.OnsetFrame[i]) + j))
        print('finish ' + imagePath + imageName + str(int(database.OnsetFrame[i]) + j))


# In[1]:


# casme2 = pd.read_csv('./database/CASME2/casme2.csv')
# segmentAll(casme2, 'CASME2', ['happiness', 'disgust', 'surprise', 'fear'])
# casme = pd.read_csv('./database/CASME/casme.csv')
# segmentAll(casme, 'CASME', ['happiness', 'disgust', 'surprise', 'fear'])
# samm = pd.read_csv('./database/SAMM/samm.csv')
# segmentAll(samm, 'SAMM', ['Happiness', 'Disgust', 'Surprise', 'Fear'])


# In[ ]:




