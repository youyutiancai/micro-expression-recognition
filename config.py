from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.applications import InceptionResNetV2, VGG19
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dropout, Concatenate, Dense, Average, Dot
from tensorflow.keras.layers import MaxPool2D, Conv2D, Add, ReLU, Lambda, Conv3D, MaxPool3D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization


from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import cv2
from sklearn.model_selection import train_test_split

def generateDataset(emotion_data, landmark):
    onsetFrame = []
    apexFrame = []
    gridHeightNum = 32
    gridWidthNum = 32
    images = []
    labels = []
    gridDiffTem = np.zeros([gridHeightNum, gridWidthNum]).astype(int)
    r = 0
    c = 0
    for i in range(len(emotion_data)):
        if emotion_data.current[i] == emotion_data.onset[i]:
            onsetFrame = cv2.imread(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
            onsetFrame = cv2.cvtColor(onsetFrame, cv2.COLOR_BGR2GRAY)
            onsetFrame = onsetFrame.astype(int)
            r = onsetFrame.shape[0]
            c = onsetFrame.shape[1]
        if emotion_data.current[i] == emotion_data.apex[i]:
            apexFrame = cv2.imread(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
            apexFrame = cv2.cvtColor(apexFrame, cv2.COLOR_BGR2GRAY)
            apexFrame = apexFrame.astype(int)
            pxMoved = findMove(onsetFrame, apexFrame)
            xDiff = pxMoved[0]
            yDiff = pxMoved[1]

            if xDiff < 0 and yDiff < 0:
                gridDiffTem = maxPxGrid(apexFrame[: r + xDiff, : c + yDiff], gridHeightNum, gridWidthNum) 
                - maxPxGrid(onsetFrame[-xDiff : r, -yDiff : c], gridHeightNum, gridWidthNum)
            if xDiff < 0 and yDiff >= 0:
                gridDiffTem = maxPxGrid(apexFrame[: r + xDiff, yDiff : c], gridHeightNum, gridWidthNum) 
                - maxPxGrid(onsetFrame[-xDiff : r, : c - yDiff], gridHeightNum, gridWidthNum)
            if xDiff >= 0 and yDiff < 0:
                gridDiffTem = maxPxGrid(apexFrame[xDiff : r, : c + yDiff], gridHeightNum, gridWidthNum) 
                - maxPxGrid(onsetFrame[: r - xDiff, -yDiff : c], gridHeightNum, gridWidthNum)
            if xDiff >= 0 and yDiff >= 0:
                gridDiffTem = maxPxGrid(apexFrame[xDiff : r, yDiff : c], gridHeightNum, gridWidthNum) 
                - maxPxGrid(onsetFrame[: r - xDiff, : c - yDiff], gridHeightNum, gridWidthNum)
            if images == []:
                images = [gridDiffTem]
            else:
                images = images + [gridDiffTem]
            if (emotion_data.emotion[i] == 'happiness' or emotion_data.emotion[i] == 'Happiness'):
#                 print(emotion_data.emotion[i])
                labels = labels + [1]
            if (emotion_data.emotion[i] == 'fear' or emotion_data.emotion[i] == 'Fear'):
#                 print(emotion_data.emotion[i])
                labels = labels + [2]
            if (emotion_data.emotion[i] == 'surprise' or emotion_data.emotion[i] == 'Surprise'):
#                 print(emotion_data.emotion[i])
                labels = labels + [3]
            if (emotion_data.emotion[i] == 'disgust' or emotion_data.emotion[i] == 'Disgust'):
#                 print(emotion_data.emotion[i])
                labels = labels + [4]
    print('finish')
    return images, labels

def normalize(images, outputNum):
    imageNum = images.shape[0]
    framePerImg = (imageNum - 1) / (outputNum + 1)
    previousFrame = images[0]
    result = np.zeros([outputNum, images[0].shape[0], images[0].shape[1]])
    for i in range(outputNum):
        currentImg = (i + 1) * framePerImg
        low = images[int(np.floor(currentImg))]
        high = images[int(np.ceil(currentImg))]
#         print(currentImg - np.floor(currentImg))
        currentNew = low + (high - low) * (currentImg - np.floor(currentImg))
        result[i] = currentNew - previousFrame
        previousFrame = currentNew
    return result

def regenerateVideo(landmark, outputNum, gridHeightNum, gridWidthNum):
    emotion_data = pd.read_csv('./database/segmented_parts.csv')
    i = 0
    j = 0
    videoNum = 0
    videos = np.zeros([emotion_data.shape[0], outputNum, gridHeightNum, gridWidthNum])
    video = np.zeros([500, gridHeightNum, gridWidthNum])
    labels = []
    while i < emotion_data.shape[0]:
        image = cv2.imread(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
#         print(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         print(image[:6, -6:])
        frame = maxPxGrid(image, gridHeightNum, gridWidthNum)
#         print(frame)
#         print(j)
        video[j] = frame
        j += 1
        if emotion_data.apex[i] == emotion_data.current[i]:
            video = video[:j]
            videos[videoNum]= normalize(video, outputNum)
            videoNum += 1
            video = np.zeros([500, gridHeightNum, gridWidthNum])
            j = 0
            if (emotion_data.emotion[i] == 'happiness' or emotion_data.emotion[i] == 'Happiness'):
                labels = labels + [1]
            if (emotion_data.emotion[i] == 'fear' or emotion_data.emotion[i] == 'Fear'):
                labels = labels + [2]
            if (emotion_data.emotion[i] == 'surprise' or emotion_data.emotion[i] == 'Surprise'):
                labels = labels + [3]
            if (emotion_data.emotion[i] == 'disgust' or emotion_data.emotion[i] == 'Disgust'):
                labels = labels + [4]
        i += 1
    print('finish')
    return videos[:videoNum], np.array(labels)

def regenerateImage(landmark, outputNum_wh, gridHeightNum, gridWidthNum):
    videos, labels = regenerateVideo(landmark, outputNum_wh * outputNum_wh, gridHeightNum, gridWidthNum)
    images = np.zeros([videos.shape[0], gridHeightNum * outputNum_wh, gridWidthNum * outputNum_wh])
    image = np.zeros([gridHeightNum * outputNum_wh, gridWidthNum * outputNum_wh])
    for m in range(videos.shape[0]):
        for i in range(outputNum_wh):
            for j in range(outputNum_wh):
                image[i * gridHeightNum : (i + 1) * gridHeightNum, 
                      j * gridWidthNum : (j + 1) * gridWidthNum] = videos[m][i * outputNum_wh + j]
        images[m] = image
    return images, labels

def findMove(old_frame, new_frame):
    n = 7
    result = np.zeros([n, n])
    nRange = int((n - 1) / 2)
    r = min([old_frame.shape[0], new_frame.shape[0]])
    c = min([old_frame.shape[1], new_frame.shape[1]])
    for i in range(n):
        i = i - nRange
        for j in range(n):
            j = j - nRange
            if i < 0 and j < 0:
                result[i + nRange][j + nRange] = (new_frame[: r + i, : c + j] - old_frame[-i : r, -j : c]).std().round(3)
            if i < 0 and j >= 0:
                result[i + nRange][j + nRange] = (new_frame[: r + i, j : c] - old_frame[-i : r, : c - j]).std().round(3)
            if i >= 0 and j < 0:
                result[i + nRange][j + nRange] = (new_frame[i : r, : c + j] - old_frame[: r - i, -j : c]).std().round(3)
            if i >= 0 and j >= 0:
                result[i + nRange][j + nRange] = (new_frame[i : r, j : c] - old_frame[: r - i, : c - j]).std().round(3)
    return (int(result.argmin() / n) - nRange, int(result.argmin() % n) - nRange)

def maxPxGrid(frame, gridHeightNum, gridWidthNum):
    (h, w) = frame.shape
    m = math.floor(h / gridHeightNum)
    n = math.floor(w / gridWidthNum)
    h_low = (m + 1) * gridHeightNum - h
    w_low = (n + 1) * gridWidthNum - w
    low_h = 0
    high_h = 0
    low_w = 0
    high_w = 0
#     print(h, w)
    maxGrids = np.zeros([gridHeightNum, gridWidthNum]).astype(int)
    newGrid = np.zeros([(m + 1) * gridHeightNum, (n + 1) * gridWidthNum])
    newGrid[: h, : w] = frame
    for i in range(gridHeightNum):
        low_h = m * i
        high_h = m * (i + 1)
        if i > h_low:
            low_h = m * h_low + (m + 1) * (i - h_low)
        if i >= h_low:
            high_h = m * h_low + (m + 1) * (i + 1 - h_low)
        for j in range(gridWidthNum):
            low_w = n * j
            high_w = n * (j + 1)
            if j > w_low:
                low_w = n * w_low + (n + 1) * (j - w_low)
            if j >= w_low:
                high_w = n * w_low + (n + 1) * (j + 1 - w_low)
            maxGrids[i][j] = int(newGrid[low_h : high_h, low_w : high_w].max())
    return maxGrids

def normalize(images, outputNum):
    imageNum = images.shape[0]
    framePerImg = (imageNum - 1) / (outputNum + 1)
    previousFrame = images[0]
    result = np.zeros([outputNum, images[0].shape[0], images[0].shape[1]])
    for i in range(outputNum):
        currentImg = (i + 1) * framePerImg
        low = images[int(np.floor(currentImg))]
        high = images[int(np.ceil(currentImg))]
#         print(currentImg - np.floor(currentImg))
        currentNew = low + (high - low) * (currentImg - np.floor(currentImg))
        result[i] = currentNew - previousFrame
        previousFrame = currentNew
    return result

def regenerateVideo(landmark, outputNum, gridHeightNum, gridWidthNum):
    emotion_data = pd.read_csv('./database/segmented_parts.csv')
    i = 0
    j = 0
    videoNum = 0
    videos = np.zeros([emotion_data.shape[0], outputNum, gridHeightNum, gridWidthNum])
    video = np.zeros([500, gridHeightNum, gridWidthNum])
    labels = []
    while i < emotion_data.shape[0]:
        image = cv2.imread(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
#         print(emotion_data.path[i] + emotion_data.image[i] + str(emotion_data.current[i]) + landmark + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         print(image[:6, -6:])
#         print(image.shape)
        frame = maxPxGrid(image, gridHeightNum, gridWidthNum)
#         print(frame)
#         print(j)
        video[j] = frame
        j += 1
        if emotion_data.apex[i] == emotion_data.current[i]:
            video = video[:j]
            videos[videoNum]= normalize(video, outputNum)
            videoNum += 1
            video = np.zeros([500, gridHeightNum, gridWidthNum])
            j = 0
            if (emotion_data.emotion[i] == 'happiness' or emotion_data.emotion[i] == 'Happiness'):
                labels = labels + [1]
            if (emotion_data.emotion[i] == 'fear' or emotion_data.emotion[i] == 'Fear'):
                labels = labels + [2]
            if (emotion_data.emotion[i] == 'surprise' or emotion_data.emotion[i] == 'Surprise'):
                labels = labels + [3]
            if (emotion_data.emotion[i] == 'disgust' or emotion_data.emotion[i] == 'Disgust'):
                labels = labels + [4]
        i += 1
    
    print('finish')

    return videos[:videoNum], np.array(labels)

def regenerateImage(landmark, outputNum_wh, gridHeightNum, gridWidthNum):
    videos, labels = regenerateVideo(landmark, outputNum_wh * outputNum_wh, gridHeightNum, gridWidthNum)
    images = np.zeros([videos.shape[0], gridHeightNum * outputNum_wh, gridWidthNum * outputNum_wh])
    image = np.zeros([gridHeightNum * outputNum_wh, gridWidthNum * outputNum_wh])
    for m in range(videos.shape[0]):
        for i in range(outputNum_wh):
            for j in range(outputNum_wh):
                image[i * gridHeightNum : (i + 1) * gridHeightNum, 
                      j * gridWidthNum : (j + 1) * gridWidthNum] = videos[m][i * outputNum_wh + j]
        images[m] = image
    return images, labels

def build_SimpleNet(input_shape, output_shape, outputsize_firstLayer, outputsize_secondLayer, outputsize_dense):
    
    input_layer = Input(shape=input_shape, dtype='float32')
    
    conv_layer = Conv2D(outputsize_firstLayer, kernel_size=3, padding='same', activation='relu')(input_layer)
    max_pool = MaxPool2D(pool_size=2, strides=2)(conv_layer)
    norm = BatchNormalization()(conv_layer)    
    dropout = Dropout(0.5)(norm)
    
    conv_layer = Conv2D(outputsize_secondLayer, kernel_size=3, padding='same', activation='relu')(dropout)
    max_pool = MaxPool2D(pool_size=2, strides=2)(conv_layer)
    norm = BatchNormalization()(conv_layer)    
    dropout_base = Dropout(0.5)(norm)
    
    flatten = Flatten()(dropout_base)
    dense = Dense(outputsize_dense, activation='relu')(flatten)
    output = Dense(output_shape, activation='softmax')(dense)
    model = Model(inputs=[input_layer], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',  #(learning_rate=0.00005)',
               metrics=['accuracy'])
    
    
    return model

def build_SimpleNet3D(input_shape, output_shape, outputsize_firstLayer, outputsize_secondLayer, outputsize_dense):
    
    input_layer = Input(shape=input_shape, dtype='float32')
    
    conv_layer = Conv3D(outputsize_firstLayer, kernel_size=3, padding='same', activation='relu')(input_layer)
    max_pool = MaxPool3D(pool_size=(1, 3, 3), strides=2)(conv_layer)
    norm = BatchNormalization()(max_pool)    
    dropout = Dropout(0.5)(norm)
    
    conv_layer = Conv3D(outputsize_secondLayer, kernel_size=3, padding='same', activation='relu')(dropout)
    max_pool = MaxPool3D(pool_size= (1, 3, 3), strides=2)(conv_layer)
    norm = BatchNormalization()(max_pool)    
    dropout_base = Dropout(0.5)(norm)
    
    flatten = Flatten()(dropout_base)
    dense = Dense(outputsize_dense, activation='relu')(flatten)
    output = Dense(output_shape, activation='softmax')(dense)
    model = Model(inputs=[input_layer], outputs=[output])
#     model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='Adam',  #(learning_rate=0.00005)',
#                metrics=['accuracy'])
    
    
    return model

def build_SimpleNet_noOutput(input_shape, output_shape, outputsize_firstLayer, outputsize_secondLayer, outputsize_dense):
    
    input_layer = Input(shape=input_shape, dtype='float32')
    
    conv_layer = Conv3D(outputsize_firstLayer, kernel_size=3, padding='same', activation='relu')(input_layer)
    max_pool = MaxPool3D(pool_size=(1, 3, 3), strides=2)(conv_layer)
    norm = BatchNormalization()(conv_layer)    
#     dropout = Dropout(0.5)(norm)
    
    conv_layer = Conv3D(outputsize_secondLayer, kernel_size=3, padding='same', activation='relu')(dropout)
    max_pool = MaxPool3D(pool_size= (1, 3, 3), strides=2)(conv_layer)
    norm = BatchNormalization()(conv_layer)    
#     dropout_base = Dropout(0.5)(norm)
    
    flatten = Flatten()(dropout_base)
#     dense = Dense(outputsize_dense, activation='relu')(flatten)
    output = Dense(output_shape, activation='softmax')(dense)
    model = Model(inputs=[input_layer], outputs=[dense])
#     model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='Adam',  #(learning_rate=0.00005)',
#                metrics=['accuracy'])
    
    
    return model