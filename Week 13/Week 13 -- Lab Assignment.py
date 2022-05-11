
'''
Lab 11
Alex Lux
Rene Escamilla
'''


##########Part 1- Image Preprocessing ###########

'''
    1)  pick an image from "img" folder (available under "week 13" module)
    
        a) load the image using "OpenCV" 
        b) plot the RGB image
        c) change the image to a grey-scale image (by "OpenCV") and plot it 
        d) rotate the image (by "OpenCV") and plot it
        e) flip the image vertically (by "OpenCV" ) and plot it
        f) find the size of the image and print it
        g) resize the image and plot it (reduce its height to 50% of its original and width to 50% of its original.)
        h) save the resulted image in part (g)      

'''


import cv2 as cv
# ------- PART A -------
cat_img = cv.imread('img/cat.jpeg')
# cv.imshow('Cat', cat_img)
# cv.waitKey(0)

import matplotlib.pyplot as plt
# ------- PART B -------
plt.imshow(cat_img)
plt.show()

# ------- PART C -------
cat_img_grayscale = cv.cvtColor(cat_img, cv.COLOR_BGR2GRAY)
plt.imshow(cat_img_grayscale)
plt.show()

# ------- PART D -------
# 90degree
grayscale_image_90_clockwise = cv.rotate(cat_img_grayscale, cv.cv2.ROTATE_90_CLOCKWISE)
# 180 degrees
grayscale_image_180 = cv.rotate(cat_img_grayscale, cv.ROTATE_180)
# 270 degrees
grayscale_image_90_counter_clockwise = cv.rotate(cat_img_grayscale, cv.ROTATE_90_COUNTERCLOCKWISE)
plt.imshow(grayscale_image_90_counter_clockwise)
plt.show()

# ------- PART E -------
greyscale_vertical_flip = cv.flip(grayscale_image_90_counter_clockwise, 0)
plt.imshow(greyscale_vertical_flip)
plt.show()

# ------- PART F -------
height, width, channels = cat_img.shape # only two channels in grey scale
print(f"IMAGE SIZE: Height: {height}, Width: {width}, Channels: {channels}")

# ------- PART G -------

print('Original Dimensions : ',cat_img.shape)
 
scale_percent = 50 # percent of original size
width = int(cat_img.shape[1] * scale_percent / 100)
height = int(cat_img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv.resize(cat_img, dim, interpolation = cv.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
plt.imshow(resized)
plt.show()

# ------- PART H -------

cv.imwrite('img/cat_resized.jpg', resized)

########## Part 2- Train NN ###########
'''
    1)  from Keras.datasets import mnist  (handwritten digit recognition task)
    2)  Use a simple NN (with 2-3 FC layers, each with 5-30 nodes) in Keras. Try to tune the hyperparameters using a validation set (e.g., set validation_split=0.3). Print the performance of your trained model on the test set.

'''
import tensorflow
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()
print(x_train[0])

from tensorflow.keras.utils import normalize

x_train2 = normalize(x_train, axis=1)
x_test2 = normalize(x_test, axis=1)

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# network parameters
batch_size = 30
dropout = 0.45

model = Sequential()
model.add(Flatten())
model.add(Dense(30, input_dim=input_size, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train2, y_train, epochs=3, validation_split=0.3 , batch_size=batch_size)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(x_test2, y_test)
print('Accuracy: %.2f' % (accuracy*100))

from tensorflow.keras.models import load_model

model.save('mnist.model')
new_model = load_model('mnist.model')
predictions = new_model.predict([x_test2])


'''
    3)  Use a simple CNN (2 convolution layers (32 filters, filter_size=(3, 3)), 1 pooling layer (pool_size=(3, 3)), and 1 FC layers) in Keras. Try to tune the hyperparameters using a validation set (e.g., set validation_split=0.3). Print the performance of your trained model on the test set.
'''
cnn_model = Sequential()

from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten

print(x_train.shape)

X_train = x_train.reshape(list(x_train.shape) + [1])    # (60000, 28, 28, 1)

cnn_model.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3), input_shape=(X_train.shape[1:])))
cnn_model.add(MaxPooling2D(pool_size=(3,3)))

cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='softmax'))
cnn_model.add(MaxPooling2D(pool_size=(3,3)))
cnn_model.add(Flatten())

cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.utils import to_categorical

import numpy as np

y_train2 = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test2 = np.asarray(y_test).astype('float32').reshape((-1,1))


history = cnn_model.fit(X_train, y_train2, epochs=3, validation_split=0.3 , batch_size=batch_size)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, accuracy = cnn_model.evaluate(x_test, y_test2)
print('Accuracy: %.2f' % (accuracy*100))

'''
    4)  Repeat Q2: this time, use a dropout layer (after the last hidden layer) in your network. 
'''
model2 = Sequential()
model2.add(Flatten())
model2.add(Dense(30, input_dim=input_size, activation='relu'))
model2.add(Dense(30, activation='relu'))
model2.add(Dropout(dropout))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model2.fit(x_train2, y_train, epochs=3, validation_split=0.3 , batch_size=batch_size)

# list all data in history
print(history2.history.keys())
# summarize history for accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, accuracy = model2.evaluate(x_test2, y_test)
print('Accuracy: %.2f' % (accuracy*100))

'''
    5)  Repeat Q3: this time, use a dropout layer (after the pooling layer) in your network. 
'''
cnn_model2 = Sequential()
cnn_model2.add(Conv2D(filters=32, activation='relu', kernel_size=(3,3), input_shape=(X_train.shape[1:])))
cnn_model2.add(MaxPooling2D(pool_size=(3,3)))

cnn_model2.add(Conv2D(filters=32, kernel_size=(3,3), activation='softmax'))
cnn_model2.add(MaxPooling2D(pool_size=(3,3)))
cnn_model2.add(Dropout(dropout))
cnn_model2.add(Flatten())

cnn_model2.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

history = cnn_model2.fit(X_train, y_train2, epochs=3, validation_split=0.3 , batch_size=batch_size)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

loss, accuracy = cnn_model2.evaluate(x_test, y_test2)
print('Accuracy: %.2f' % (accuracy*100))