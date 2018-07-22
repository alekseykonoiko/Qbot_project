import os
import numpy as np
from numpy.random import seed #fix random seed for reproducibility (numpy)
seed(1)
from tensorflow import set_random_seed # fix random seed for reproducibility (tensorflow backend)
set_random_seed(2)
from numpy import array
from random import randint
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
from numpy import genfromtxt
import matplotlib.pyplot as plt
import keras
from keras import initializers
from keras import layers
from keras.layers import *
from keras.utils import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

#### Load Data1 ####

npzfile = np.load('training_data.npz')
npzfile.files
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_test']
y_test = npzfile['y_test']

#Indian Pines

#Pavia University


# Data Parameters
# h= height 
# w= width
# d= depth
# s= number of samples

#### Hyperparameters ####
batch_size = 10 #40
num_classes = 8
epochs = 1

#### CNN structure (Functional API Model Style) ####

## Uncomment initializer to be used 
#initializer = keras.initializers.Ones()
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=True)
#initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
#initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
#initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
#initializer = keras.initializers.Orthogonal(gain=2, seed=True)
#initializer = keras.initializers.lecun_uniform(seed=True)
#initializer = keras.initializers.glorot_normal(seed=None)
#initializer = keras.initializers.glorot_uniform(seed=None)
#initializer = keras.initializers.he_normal(seed=True)
#initializer = keras.initializers.lecun_normal(seed=None)
#initializer = keras.initializers.he_uniform(seed=None)


input1 = Input(shape=(7,7,200,1))

conv1 = Conv3D(16, kernel_size=(3, 3, 11), strides=(1, 1, 3), padding='same', activation='relu')(input1)

conv2_1 = Conv3D(16, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(conv1)
conv2_2 = Conv3D(16, kernel_size=(1, 2, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv1)
conv2_3 = Conv3D(16, kernel_size=(1, 1, 5), strides=(1, 1, 1), padding='same',activation='relu')(conv1)
conv2_4 = Conv3D(16, kernel_size=(1, 1, 11), strides=(1, 1, 1), padding='same',activation='relu')(conv1)

sum1 = concatenate([conv2_1, conv2_2,conv2_3,conv2_4], axis=3)

conv3_1 = Conv3D(16, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(sum1)
conv3_2 = Conv3D(16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='same', activation='relu')(sum1)
conv3_3 = Conv3D(16, kernel_size=(1, 1, 5), strides=(1, 1, 1), padding='same',activation='relu')(sum1)
conv3_4 = Conv3D(16, kernel_size=(1, 1, 11), strides=(1, 1, 1), padding='same',activation='relu')(sum1)

sum2 = concatenate([conv3_1, conv3_2,conv3_3,conv3_4], axis=3)

conv4 = Conv3D(16, kernel_size=(2, 2, 3), strides=(1, 1, 1), padding='same',activation='relu')(sum2)
pool = MaxPooling3D((2, 2, 3), strides=(2, 2, 3), padding='same')(conv4)
drop = SpatialDropout3D(0.6)(pool)

flat = Flatten()(drop)
output = Dense(num_classes, activation='softmax')(flat)

## initiate optimizer

opt0 = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-4)

opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

opt2 = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

opt3=keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01)

## Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy', 
              
              optimizer=opt3, 
              
              metrics=['accuracy'])





## Scale 0-255 bands range into float 0-1
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

#x_train /= 100
#x_test /= 100

print(model.summary()) # summarize layers

x_train = x_train.reshape(840, 32, 32, 288)
# y_train = y_train.reshape(840, 6, 1, 1, 1)
x_test = x_test.reshape(120, 32, 32, 288)
# y_test = y_test.reshape(120, 6, 1, 1, 1)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

# checkpoint
filepath="logs/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#TensorBoard
tensorflow = keras.callbacks.TensorBoard(log_dir='logs')
callbacks_list = [checkpoint, tensorflow]

#np.random.seed(seed)
cnn = model.fit(x_train, y_train,
          
              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test),
          
              verbose=1,

              callbacks=callbacks_list,

              shuffle=True)


#### Save model ####

model.save_weights('H_K_2016_trained_model_weights.h5')
# model_path = os.path.join(save_dir, model_name)

# model.save(model_path)

# print('Saved trained model at %s ' % model_path)

#### Model testing ####
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])