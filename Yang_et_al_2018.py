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

npzfile = np.load('training_data.npz')
npzfile.files
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_test']
y_test = npzfile['y_test']



#### Hyperparameters ####
batch_size = 4
num_classes = 6
epochs = 1

#### CNN structure (Functional API Model Style) ####

## Uncomment initializer to be used 
#initializer = keras.initializers.Ones()
#initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=True)
#initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
#initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
#initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
initializer = keras.initializers.Orthogonal(gain=2, seed=True)
#initializer = keras.initializers.lecun_uniform(seed=True)
#initializer = keras.initializers.glorot_normal(seed=None)
#initializer = keras.initializers.glorot_uniform(seed=None)
#initializer = keras.initializers.he_normal(seed=True)
#initializer = keras.initializers.lecun_normal(seed=None)
#initializer = keras.initializers.he_uniform(seed=None)


input1 = Input(shape=(32,32,288,1))
input2 = Input(shape=(14,14,288,1))

spatial = Conv3D(288, kernel_size=(3, 3, 3), padding='same', activation='relu')(input1)
spatial2 = Conv3D(286, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial)
spatial3 = Conv3D(284, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial2)
spatial4 = Conv3D(282, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial3)
spatial5 = Conv3D(280, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial4)
spatial6 = Conv3D(278, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial5)
spatial7 = Conv3D(276, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial6)
spatial8 = Conv3D(274, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial7)
spatial9 = Conv3D(272, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial8)
spatial10 = Conv3D(270, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(spatial9)


spectral = Conv3D(288, kernel_size=(1, 1, 5), padding='same', activation='relu')(input2)
spectral2 = Conv3D(284, kernel_size=(1, 1, 5), strides=(1, 1, 1), activation='relu')(spectral)
spectral3 = Conv3D(280, kernel_size=(1, 1, 5), strides=(1, 1, 1), activation='relu')(spectral2)
spectral4 = Conv3D(275, kernel_size=(1, 1, 6), strides=(1, 1, 1), activation='relu')(spectral3)
spectral5 = Conv3D(270, kernel_size=(1, 1, 6), strides=(1, 1, 1), activation='relu')(spectral4)

# concat = merge([UpSampling3D(size=(288, 1, 1))(spatial), spectral], mode='concat', concat_axis=1)
concat = concatenate([spatial10, spectral5], axis=3)

concat2 = Conv3D(270, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(concat)
concat3 = Conv3D(268, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu')(concat2)
concat4 = Conv3D(134, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(concat3)
concat5 = Conv3D(132, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu')(concat4)
concat6 = Conv3D(66, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(concat5)
concat7 = Conv3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='relu')(concat6)
concat8 = Conv3D(32, kernel_size=(1, 1, 3), strides=(1, 1, 1), activation='relu')(concat7)


flat = Flatten()(concat8)
# x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
# x = Dropout(0.2)(x)
# x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
# x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(flat)
model = Model(inputs=(input1,input2), outputs=output)


## initiate RMSprop optimizer

#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-4)

#opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

opt2 = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

## Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy', 
              
              optimizer=opt, 
              
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