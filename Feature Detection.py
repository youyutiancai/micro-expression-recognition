#!/usr/bin/env python
# coding: utf-8

# In[129]:


from imutils import face_utils
import numpy as np
import imutils
import cv2
import dlib
import collections
import matplotlib.pyplot as plt


# In[130]:


# construct the argument parser and parse the arguments
shape_predictor = "shape_predictor_68_face_landmarks.dat"
happyimages = ["images/happy.png", "images/happy1.jpg", "images/happy2.jpg", "images/happy3.jpg", "images/happy4.jpg", "images/happy5.jpg", "images/happy6.jpg", "images/happy7.jpg"]
sadimages = ["images/sad.jpg", "images/sad1.jpg", "images/sad2.jpg", "images/sad3.png", "images/sad4.jpg", "images/sad5.jpg", "images/sad6.jpg", "images/sad7.jpg"]


# In[131]:


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


# In[132]:


def readImage(image):  
   # load the input image, resize it, and convert it to grayscale
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
           clone = image.copy()
           cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 0, 255), 2)

           # loop over the subset of facial landmarks, drawing the
           # specific face part
           for (x, y) in shape[i:j]:
               cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

           # extract the ROI of the face region as a separate image
#             (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
#             roi = image[y:y + h, x:x + w]
#             roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

           #show the particular face part
#             cv2.imshow("ROI", roi)
#             cv2.imshow("Image", clone)
#             cv2.waitKey(0)
   return faceparts


# In[133]:


def getSlope(faceparts, partofmouth):
    mouthpoints = faceparts[partofmouth]
    corner1 = mouthpoints[0]
    corner2 = mouthpoints[0]
    for point in mouthpoints:
        if(point[0] < corner1[0]):
            corner1 = point
        if(point[0] > corner2[0]):
            corner2 = point
    middle = mouthpoints[0]
    middlenum = corner1[0] + (corner2[0]-corner1[0])/2
    distance = 100000000000000
    for point in mouthpoints:
        if(abs(point[0]-middlenum) <= (distance+7)):
            if(point[1] > middle[1]):
                middle = point 
                distance = abs(point[0]-middlenum)
    slope = (corner1[1]-middle[1])/(corner1[0]-middle[0])
    slope2 = (corner2[1]-middle[1])/(corner2[0]-middle[0])
    if(slope > 0):
        avgslope = (abs(slope)+abs(slope2))/2
    else: 
        avgslope = -1*(abs(slope)+abs(slope2))/2
    return avgslope


# In[134]:


def getFlatSlope(faceparts, partofmouth):
    mouthpoints = faceparts[partofmouth]
    corner1 = mouthpoints[0]
    corner2 = mouthpoints[0]
    for point in mouthpoints:
        if(point[0] < corner1[0]):
            corner1 = point
        if(point[0] > corner2[0]):
            corner2 = point
    middle = mouthpoints[0]
    middlenum = corner1[0] + (corner2[0]-corner1[0])/2
    distance = 100000000000000
    for point in mouthpoints:
        if(abs(point[0]-middlenum) <= (distance+7)):
            if(point[1] < middle[1]):
                middle = point 
                distance = abs(point[0]-middlenum)
    slope = (corner1[1]-middle[1])/(corner1[0]-middle[0])
    slope2 = (corner2[1]-middle[1])/(corner2[0]-middle[0])
    if(slope > 0):
        avgslope = (abs(slope)+abs(slope2))/2
    else: 
        avgslope = -1*(abs(slope)+abs(slope2))/2
    return avgslope


# In[153]:


def getEyebrowDistance(faceparts, eye, eyebrow):         
    eyepoints = faceparts[eye]
    top = eyepoints[0]
    for point in eyepoints:
        if(point[0] < top[0]):
            top = point
    eyepoints = faceparts[eyebrow]
    distancetoeyepoint = 10000000
    closestpoint = eyepoints[0]
    for point in eyepoints:
        if(abs(point[0] - top[0]) < distancetoeyepoint):
            distancetoeyepoint = abs(point[0] - top[0])
            closestpoint = point
    return abs(closestpoint[1]-top[1])  


# In[284]:


def getMouthValue(faceparts, image):
    mouthslope = getSlope(faceparts, 'mouth')
    jawslope = getSlope(faceparts, 'jaw')
    mouthsecondslope = getFlatSlope(faceparts, 'mouth')
    return mouthslope/(jawslope*mouthsecondslope)


# In[285]:


def getEyeRatio(faceparts, eye):
    eyepoints = faceparts[eye]
    corner1 = eyepoints[0]
    corner2 = eyepoints[0]
    top = eyepoints[0]
    bottom = eyepoints[0]
    for point in eyepoints:
        if(point[0] < corner1[0]):
            corner1 = point
        if(point[0] > corner2[0]):
            corner2 = point
        if(point[1] < top[1]):
            top = point
        if(point[1] > bottom[1]):
            bottom = point
    return (corner2[0]-corner1[0])/(bottom[1]-top[1])


# In[286]:


def getEyeValue(faceparts, image):
    ratio = getEyeRatio(faceparts, 'right_eye')
    ratio2 = getEyeRatio(faceparts, 'left_eye')
    distance1 = getEyebrowDistance(faceparts, 'right_eye', 'right_eyebrow')  
    distance2 = getEyebrowDistance(faceparts, 'left_eye', 'left_eyebrow') 
    avgratio = 5*((ratio/ratio2)/(distance1/distance2)) + 10*((ratio/distance1)+(ratio2/distance2))
    return avgratio


# In[287]:


happymouthvals = list()
happyeyevals = list()
for image in happyimages:
    faceparts = readImage(image)
    happymouthvals.append(getMouthValue(faceparts, image))
    happyeyevals.append(getEyeValue(faceparts, image))
print(happymouthvals)
print(happyeyevals)


# In[288]:


sadmouthvals = list()
sadeyevals = list()
for image in sadimages:
    faceparts = readImage(image)
    sadmouthvals.append(getMouthValue(faceparts, image))
    sadeyevals.append(getEyeValue(faceparts, image))
print(sadmouthvals)
print(sadeyevals)


# In[289]:


plt.plot(happymouthvals, np.zeros_like(happymouthvals) + 0, 'x')
plt.plot(sadmouthvals, np.zeros_like(sadmouthvals) + 0, 'o')
plt.show()

# In[290]:


plt.plot(happyeyevals, np.zeros_like(happyeyevals) + 0, 'x')
plt.plot(sadeyevals, np.zeros_like(sadeyevals) + 0, 'o')

