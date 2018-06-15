
#%matplotlib qt
#%matplotlib notebook
#%matplotlib inline
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

#### Load matlab 3 dimensional array into Jupiter notebook ####

## type 5,10,15 & un/trimmed correspondingly into image_<type_here>per_<type_here> to choose data ## 
mat_contents = sio.loadmat('hyperspectral_images/trimmed_data/image_5per_trimmed.mat') #loads .mat file
## type 5,10,15 & un/trimmed correspondingly into X<type_here>_<type_here> to choose data ##
image_data5=mat_contents['X5_trimmed'] #extracts data from .mat file
print ('5% water image shape', image_data5.shape)

mat_contents = sio.loadmat('hyperspectral_images/trimmed_data/image_10per_trimmed.mat') #loads .mat file
image_data10=mat_contents['X10_trimmed'] #extracts data from .mat file
print ('10% water image shape', image_data10.shape)

mat_contents = sio.loadmat('hyperspectral_images/trimmed_data/image_15per_trimmed.mat') #loads .mat file
image_data15=mat_contents['X15_trimmed'] #extracts data from .mat file
print ('15% water image shape', image_data15.shape)

#### To check data validity visually ####
from spectral import *
view = imshow(image_data5)
title_obj = plt.title('5% water image')

view = imshow(image_data10)
title_obj = plt.title('10% water image')
## Filter for saturated pixels
for i in range(0, 256):
    for j in range(0, 256):
        for k in range(0, 288):
            pixel_value = image_data10[j, i, k]
            if pixel_value > 70: # Threhold value for saturated pixel from 0 to 1000
                image_data10[j, i, k] = image_data10[j-1, i-1, k]
view = imshow(image_data10)
title_obj = plt.title('10% water image (filtered)')

view = imshow(image_data15)
title_obj = plt.title('15% water image')

#### Segmentation of picture into multiple pictures and arrangment into cifar10 dataset style ####
image_data_segments = list()
image_data_segments_test = list()
n=8 #defines number of segmentation per side (only even number)
p=32 #defines target segmented picture size
k=0 #counter for picture height segmentation

for i in range(0, n):
    for j in range(0, n):
        #Segments picture row by row
        if j<n-1:
            image_data_segment5 = image_data5[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments.append(image_data_segment5)
        else:
            image_data_segment5 = image_data5[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments_test.append(image_data_segment5)
    for j in range(0, n):
        if j<n-1:
            image_data_segment10 = image_data10[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments.append(image_data_segment10)
        else:
            image_data_segment10 = image_data10[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments_test.append(image_data_segment10)
    for j in range(0, n):
        if j<n-1:
            image_data_segment15 = image_data15[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments.append(image_data_segment15)
        else:
            image_data_segment15 = image_data15[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments_test.append(image_data_segment15)
    k += 1
    

x_train = array(image_data_segments)
x_test = array(image_data_segments_test)
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)

##  Create cifar10 style labels array 
y_labels = list()
y_labels_test = list()
label = np.array([0, 1, 2]) #sets label class value

for i in range(0, ((n*n)*3)//3):
    if i<(n-1)*n:
        y_label=label[0]
        y_labels.append(y_label)
    else:
        y_label=label[0]
        y_labels_test.append(y_label) 
for i in range(0, ((n*n)*3)//3):
    if i<(n-1)*n:
        y_label=label[1]
        y_labels.append(y_label)
    else:
        y_label=label[1]
        y_labels_test.append(y_label)    
for i in range(0, ((n*n)*3)//3):
    if i<(n-1)*n:
        y_label=label[2]
        y_labels.append(y_label)
    else:
        y_label=label[2]
        y_labels_test.append(y_label) 
    
y_train = array(y_labels)
y_train = y_train.reshape(len(x_train), 1)
y_test = array(y_labels_test)
y_test = y_test.reshape(len(x_test), 1)

print ('y_train shape:', y_train.shape)
print ('y_test shape:', y_test.shape)
#print ('Labels array:', y_train)

## hyperparameters section 
batch_size = 64
num_classes = 3
epochs = 200

## Convert class vectors to binary class matrices.

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

#print (y_train)
#print (y_test)

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

x = Conv2D(64, kernel_size=1, activation='relu', kernel_initializer=initializer)(input1)
#x = Conv2D(288, kernel_size=7, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
#x = AveragePooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.2)(x)

#x = Conv2D(288, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = Conv2D(32, kernel_size=7, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.2)(x)

x = Conv2D(32, kernel_size=5, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
#x = Conv2D(32, kernel_size=7, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
x = AveragePooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.2)(x)

# x = Conv2D(64, kernel_size=3, padding='same' ,activation='relu', kernel_initializer=initializer)(x)
# x = AveragePooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(32, activation='relu', kernel_initializer=initializer)(x)
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

np.random.seed(seed)
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

plt.xticks(np.arange(0, epochs+1, 10))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])


plt.figure(6)

plt.plot(cnn.history['loss'],'r')

plt.plot(cnn.history['val_loss'],'g')

plt.xticks(np.arange(0, epochs+1, 10))

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

