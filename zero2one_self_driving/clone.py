import sys
import csv
from tqdm import tqdm
import cv2
import datetime
import h5py
import numpy as np

#ライブラリのimport
from tensorflow.keras import Sequential,models
from tensorflow.keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def getrowsFromDrivingLogs(dataPath):
    rows = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            rows.append(row)
    return rows

def getImageArray3angle(originalImage, images):
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    images.append(image)
#     steerings.append(steering)
    
def getImagesAndSteerings(rows):
    
    images = []
    steerings = []
    
    for row in tqdm(rows):
        steering = float(row[3])
        # 左右のカメラのステアリング測定値を調整します
        parameter = 0.2 
        # このパラメータが調整用の値です。
        # 左のカメラはステアリング角度が実際よりも低めに記録されているので、少し値を足してやります。右のカメラはその逆です。
        steering_left = steering + parameter
        steering_right = steering - parameter
 
        # データセットに画像とステアリング角度を加えます
        #center
        getImageArray3angle(cv2.imread(row[0].strip()), images)
        steerings.append(steering)
        #left
        getImageArray3angle(cv2.imread(row[1].strip()), images)
        steerings.append(steering_left)
        #right
        getImageArray3angle(cv2.imread(row[2].strip()), images)
        steerings.append(steering_right)
    
    return (np.array(images), np.array(steerings))

def trainModelAndSave(model, inputs, outputs, epochs, batch_size):
    
    X_train, X_valid, y_train, y_valid = train_test_split(inputs, outputs, test_size=0.2, shuffle=False)
    #Setting model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    #Learning model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))
    #Saving model
    model.save('model.h5')
    
#essential network
def nn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

#convolutional network
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=(160, 320, 3)))
    model.add(Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model

#NVIDIA
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    
    epochs = 200
    #Make sure to set batch size within 40
    batch_size = 40
    is_dataset = True
#     is_dataset = False
    
    #When making "is_dataset" True, saving preprocessed datasets
    #When making "is_dataset" False, using preprocessed and saved datasets. Shortens time because preprocessing is not required.It must be preprocessed once.
    if is_dataset:
        print('get csv data from Drivinglog.csv')
        rows = getrowsFromDrivingLogs('data')
        print('preprocessing data...')
        inputs, outputs = getImagesAndSteerings(rows)
        
        with h5py.File('./trainingData.h5', 'w') as f:
            f.create_dataset('inputs', data=inputs)
            f.create_dataset('outputs', data=outputs)
    
    else:
        with h5py.File('./trainingData.h5', 'r') as f:
            inputs = np.array(f['inputs'])
            outputs = np.array(f['outputs'])

    print('Training data:', inputs.shape)
    print('Training label:', outputs.shape)
    
    #Specifying model
    model = nvidia_model()
    #Training and saving model
    trainModelAndSave(model, inputs, outputs, epochs, batch_size)
