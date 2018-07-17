## Cross-Domain CNN for Hyperspectral Image Classification 
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

# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'keras__trained_model.h5'

#### Load Data1 ####

npzfile = np.load('training_data_no_norm.npz')
npzfile.files
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_test']
y_test = npzfile['y_test']



#### Hyperparameters ####
batch_size = 4 #10
num_classes = 6 
epochs = 1 #100000

#### CNN structure (Functional API Model Style) ####

## Uncomment initializer to be used 
#initializer = keras.initializers.Ones()
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=True)
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


input1 = Input(shape=(32,32,288,1))

spatial0= Conv3D(128, kernel_size=(5, 5, 288), padding='same', activation='relu')(input1)
spatial0bn= BatchNormalization()(spatial0)
spatial1= Conv3D(128, kernel_size=(3, 3, 288), padding='same', activation='relu')(input1)
spatial1bn= BatchNormalization()(spatial1)
spectral = Conv3D(128, kernel_size=(1, 1, 288), padding='same', activation='relu')(input1)
spectralbn= BatchNormalization()(spectral)

# concat = merge([UpSampling3D(size=(288, 1, 1))(spatial), spectral], mode='concat', concat_axis=1)
concat = concatenate([spatial0bn, spatial1bn, spectralbn], axis=3)


l1 = Conv3D(128, kernel_size=(1, 1, 32), strides=(1, 1, 18), activation='relu')(concat)
noise1 = GaussianNoise(0.01)(l1)
noise1bn = BatchNormalization()(noise1)
l2 = Conv3D(128, kernel_size=(1, 1, 32), padding='same', activation='relu')(noise1)
noise2 = GaussianNoise(0.01)(l2)
noise2bn = BatchNormalization()(noise2)
l3 = Conv3D(128, kernel_size=(1, 1, 32), padding='same', activation='relu')(noise2)
noise3 = GaussianNoise(0.005)(l3)
noise3bn = BatchNormalization()(noise3)
add1 = add([l1, noise3bn])

l4 = Conv3D(128, kernel_size=(1, 1, 9), strides=(1, 1, 3), activation='relu')(add1)
noise4 = GaussianNoise(0.005)(l4)
noise4bn = BatchNormalization()(noise4)
l5 = Conv3D(128, kernel_size=(1, 1, 9), padding='same',activation='relu')(noise4)
noise5 = GaussianNoise(0.005)(l5)
noise5bn = BatchNormalization()(noise5)
l6 = Conv3D(128, kernel_size=(1, 1, 9), padding='same',activation='relu')(noise5)
noise6 = GaussianNoise(0.005)(l6)
noise6bn = BatchNormalization()(noise6)
add2 = add([l4, noise6bn])

l7 = Conv3D(128, kernel_size=(1, 1, 3), activation='relu')(add2)
noise7 = GaussianNoise(0.005)(l7)
noise7bn = BatchNormalization()(noise7)
drop1 = SpatialDropout3D(0.2)(noise7bn)
l8 = Conv3D(128, kernel_size=(1, 1, 3), activation='relu')(drop1)
noise8 = GaussianNoise(0.005)(l8)
noise8bn = BatchNormalization()(noise8)
drop2 = SpatialDropout3D(0.2)(noise8bn)
l9 = Conv3D(128, kernel_size=(1, 1, 3), activation='relu')(drop2)
noise9 =  GaussianNoise(0.01)(l9)


flat = Flatten()(noise9)
# x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
# x = Dropout(0.2)(x)
# x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
# x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(flat)
model = Model(inputs=input1, outputs=output)


## initiate RMSprop optimizer

opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-4)

opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

opt2 = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

## Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy', 
              
              optimizer=opt2, 
              
              metrics=['accuracy'])
## Scale 0-255 bands range into float 0-1
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 100

x_test /= 100

print(model.summary()) # summarize layers

x_train = x_train.reshape(840, 32, 32, 288, 1)
# y_train = y_train.reshape(840, 6, 1, 1, 1)
x_test = x_test.reshape(120, 32, 32, 288, 1)
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

model.save_weights('trained_model_weights.h5')
# model_path = os.path.join(save_dir, model_name)

# model.save(model_path)

# print('Saved trained model at %s ' % model_path)

#### Model testing ####
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])