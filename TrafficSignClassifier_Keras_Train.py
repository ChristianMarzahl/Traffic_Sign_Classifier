# Load pickled data
import pickle
import cv2
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('files/train_augmented_gray.p', mode='rb') as f:
    dataTrain = pickle.load(f)

with open('files/valid.p', mode='rb') as f:
    dataValid = pickle.load(f)

with open('files/test.p', mode='rb') as f:
    dataTest = pickle.load(f)

X_train, y_train = dataTrain['features'], dataTrain['labels']
X_valid, y_valid = dataValid['features'], dataValid['labels']
X_test, y_test = dataTest['features'], dataTest['labels']


X_valid = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_valid])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_test])


# preprocess data
X_train_normalized = np.array(X_train / 255.0 - 0.5 )
X_valid_normalized = np.array(X_valid / 255.0 - 0.5 )
X_test_normalized = np.array(X_test / 255.0 - 0.5 )


from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)
y_one_hot_valid = label_binarizer.fit_transform(y_valid)
y_one_hot_test = label_binarizer.fit_transform(y_test)


# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# TODO: Build the Final Test Neural Network in Keras Here
# Variante 1 
#model = Sequential()
#model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 1)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dense(43))
#model.add(Activation('softmax'))

# Variante 2 
# Epoch 20/100 val_acc: 0.9245
# Epoch 65/100 val_acc: 0.9431
model = Sequential()
model.add(Convolution2D(16, 5, 5, input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(360))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(43))
model.add(Activation('softmax'))

# Variante 3 LeNet
# http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
# Epoch 25 acc: 0.88
#model = Sequential()
## first set of CONV => RELU => POOL
#model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(32, 32, 1)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

## second set of CONV => RELU => POOL
#model.add(Convolution2D(50, 5, 5, border_mode="same"))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

## set of FC => RELU layers
#model.add(Flatten())
#model.add(Dense(500))
#model.add(Dropout(0.25))
#model.add(Activation("relu"))
 
## softmax classifier
#model.add(Dense(43))
#model.add(Activation("softmax"))

#Variante 4 
# Epoch 25 test acc: 0.91
# Epoch 100 test acc: 0.92
#model = Sequential()
## first set of CONV => RELU => POOL
#model.add(Convolution2D(20, 5, 5, border_mode="same",input_shape=(32, 32, 1)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

## second set of CONV => RELU => POOL
#model.add(Convolution2D(50, 5, 5, border_mode="same"))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

## set of FC => RELU layers
#model.add(Flatten())
#model.add(Dense(500))
#model.add(Dense(300))
#model.add(Dropout(0.25))
#model.add(Dense(100))
#model.add(Dropout(0.25))
 
## softmax classifier
#model.add(Dense(43))
#model.add(Activation("softmax"))

weights_file = 'traffic_sign.hdf5'
# load weights
#model.load_weights(weights_file)

model_checkpoint = ModelCheckpoint(weights_file, monitor='loss', save_best_only=True)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_train_normalized, y_one_hot_train, nb_epoch=100, callbacks=[model_checkpoint], shuffle=True, verbose=1, batch_size=512, validation_data=(X_valid_normalized, y_one_hot_valid))


print("Testing")

metrics = model.evaluate(X_test_normalized, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
    