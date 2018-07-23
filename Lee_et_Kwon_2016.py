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
batch_size = 10
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


input1 = Input(shape=(32,32,288))

spatial = Conv2D(128, (1, 1), padding='same', activation='relu')(input1)
spatial= Conv2D(128, (3, 3), padding='same', activation='relu')(spatial)

spectral = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input1)
spectral = Conv2D(64, (1, 1), padding='same', activation='relu')(spectral)

concat = concatenate([spatial, spectral], axis=3)

l1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(concat)
l1bn=BatchNormalization()(l1)
l2 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(l1bn)
l2bn=BatchNormalization()(l2)
l3 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(l2bn)
add1 = add([l1bn, l3])


l4 = Conv2D(128, kernel_size=(1, 1), padding='same',activation='relu')(add1)
l5 = Conv2D(128, kernel_size=(1, 1), padding='same',activation='relu')(l4)
add2 = add([l4, l5])

l7 = Conv2D(128, kernel_size=(1, 1), activation='relu')(add2)
drop1 = SpatialDropout2D(0.2)(l7)
l8 = Conv2D(128, kernel_size=(1, 1), activation='relu')(drop1)
drop2 = SpatialDropout2D(0.2)(l8)
l9 = Conv2D(128, kernel_size=(1, 1), activation='relu')(drop2)

flat = Flatten()(l9)
output = Dense(num_classes, activation='softmax')(flat)
model = Model(inputs=input1, outputs=output)

## initiate RMSprop optimizer

opt0 = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-4)

opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

opt2 = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

## Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy', 
              
              optimizer=opt2, 
              
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