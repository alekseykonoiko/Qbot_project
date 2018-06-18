# this is a testt 
#%matplotlib qt
#%matplotlib notebook
#%matplotlib inlinen
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Comment to Enable GPU
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

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras__trained_model.h5'

#### Load Data ####

npzfile = np.load('training_data.npz')
npzfile.files
x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test = npzfile['x_test']
y_test = npzfile['y_test']
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


#### Hyperparameters ####
batch_size = 32
num_classes = 6
epochs = 200

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


input1 = Input(shape=(32,32,288))

x = Conv2D(64, kernel_size=5, padding='same', activation='relu', kernel_initializer=initializer)(input1)
#x = Conv2D(288, kernel_size=7, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
#x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, kernel_size=5, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
#x = Conv2D(288, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = Conv2D(64, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
#x = Conv2D(32, kernel_size=7, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

# x = Conv2D(64, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
# x = AveragePooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu', kernel_initializer=initializer)(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input1, outputs=output)

## initiate RMSprop optimizer

opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-4)

opt1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

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
#plot_model(model, to_file='Pictures/convolutional_neural_network.png') # plot graph of CNN structure

#np.random.seed(seed)
cnn = model.fit(x_train, y_train,
          
              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test),
          
              verbose=1,

              shuffle=True)

#### Training Stats ####

plt.figure(5)

plt.plot(cnn.history['acc'],'r')

plt.plot(cnn.history['val_acc'],'g')

plt.xticks(np.arange(0, epochs+1, 25))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])


plt.figure(6)

plt.plot(cnn.history['loss'],'r')

plt.plot(cnn.history['val_loss'],'g')

plt.xticks(np.arange(0, epochs+1, 25))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.legend(['train','validation'])

plt.show()

#### Save model ####

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)

#### Model testing ####
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])

##
