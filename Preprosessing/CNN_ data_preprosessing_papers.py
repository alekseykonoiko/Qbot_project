#%matplotlib qt #console library to preview images\n
#%matplotlib notebook\n",
#%matplotlib inline\n",
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Comment to Enable GPU
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as sio


#### Load matlab 3 dimensional array into Jupiter notebook ####   
mat_contents = sio.loadmat('CNNs/Preprosessing/Indian_pines.mat') #loads .mat file
indian_pines=mat_contents['indian_pines'] #extracts data from .mat file
print ('Indian Pines image shape', indian_pines.shape)
mat_contents = sio.loadmat('CNNs/Preprosessing/Indian_pines_gt') #loads .mat file
indian_pines_gt=mat_contents['indian_pines_gt'] #extracts data from .mat file
print ('Indian Pines ground truth image shape', indian_pines_gt.shape)

mat_contents = sio.loadmat('CNNs/Preprosessing/PaviaU') #loads .mat file
pavia_university=mat_contents['paviaU'] #extracts data from .mat file
print ('Pavia University image shape', pavia_university.shape)
mat_contents = sio.loadmat('CNNs/Preprosessing/PaviaU_gt') #loads .mat file
pavia_university_gt=mat_contents['paviaU_gt'] #extracts data from .mat file
print ('Pavia University ground truth image shape', pavia_university_gt.shape)

""" mat_contents = sio.loadmat('CNNs/Preprosessing/Salinas') #loads .mat file
salinas=mat_contents['salinasU'] #extracts data from .mat file
print ('Salinas image shape', salinas.shape) """
""" mat_contents = sio.loadmat('CNNs/Preprosessing/Salinas') #loads .mat file
salinas_gt=mat_contents['salinas_gt'] #extracts data from .mat file
print ('Salinas ground truth image shape', salinas_gt.shape) """
indian_pines_data_segments = list()
indian_pines_data_segments_test = list()

for i in range(1, indian_pines_gt.shape[0]-1):
    for j in range(1, indian_pines_gt.shape[1]-1):
        if indian_pines_gt[i, j] == 1:
            indian_pines_segment = indian_pines[i - 1:i + 1, j - 1:j + 1, :]
            indian_pines_data_segments.append(indian_pines_segment)
            print ('segment detected!')
        print (i, ', ', j)


print(indian_pines.shape[1])
print(indian_pines_gt.shape[1])
print(pavia_university.shape[1])
print(pavia_university_gt.shape[1])


