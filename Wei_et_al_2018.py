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

#Salinas Scene


# Data Parameters
# h= height 
# w= width
# d= depth
# s= number of samples #200

#### Hyperparameters ####
batch_size = 10
num_classes = K #9
epochs = 1 #100

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


input1 = Input(shape=(2*3,3,103,1)) #input is normalised from 0 t 1

# spatial = Conv2D(128, (1, 1), padding='same', activation='relu')(input1)
# spatial= Conv2D(128, (3, 3), padding='same', activation='relu')(spatial)

# spectral = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input1)
# spectral = Conv2D(64, (1, 1), padding='same', activation='relu')(spectral)

# concat = concatenate([spatial, spectral], axis=3)

l1 = Conv3D(6, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(input1)
l2 = Conv3D(6, kernel_size=(3, 1, 8), strides=(1, 1, 3), padding='same', activation='relu')(l1)
l3 = Conv3D(12, kernel_size=(1, 2, 3), strides=(1, 1, 1), padding='same', activation='relu')(l2)
l4 = Conv3D(24, kernel_size=(3, 1, 3), strides=(1, 1, 2), padding='same',activation='relu')(l3)
l5 = Conv3D(48, kernel_size=(2, 1, 3), strides=(1, 1, 1), padding='same',activation='relu')(l4)
l6 = Conv3D(48, kernel_size=(1, 2, 3), strides=(1, 1, 2), padding='same', activation='relu')(l5)
l7 = Conv3D(96, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='same', activation='relu')(l6)
l8 = Conv3D(96, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='same', activation='relu')(l7)
l9 = Conv3D(10, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(l8)

output = Conv3D(K, kernel_size=(K+1, 1, 1), padding='same', activation='softmax')(l9) # output size C+1 *1*1
model = Model(inputs=input1, outputs=output)

## initiate optimizer

opt0 = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-4)

opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

opt2 = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

model.compile(loss='categorical_crossentropy', 
              
              optimizer=opt1, 
              
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