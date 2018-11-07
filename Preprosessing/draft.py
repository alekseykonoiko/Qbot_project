#### Segmentation of picture into multiple pictures and arrangment into cifar10 dataset style ####
image_data_segments = list()
image_data_segments_test = list()


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
            sand_0_segment =  sand_0[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments.append(sand_0_segment)
        else:
            sand_0_segment =  sand_0[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments_test.append(sand_0_segment)
    
    for j in range(0, n):
        if j<n-1:
            sand_5_segment = sand_5[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments.append(sand_5_segment)
        else:
            sand_5_segment = sand_5[p*k:p*(k+1),p*j:p*(j+1),:]
            image_data_segments_test.append(sand_5_segment)
    k += 1
    
x_train = array(image_data_segments)
x_test = array(image_data_segments_test)
print ('x_train shape:', x_train.shape)
print ('x_test shape:', x_test.shape)

##  Create cifar10 style labels array 
y_labels = list()
y_labels_test = list()
label = np.array([0, 1, 2, 3, 4, 5])  #sets label class value

for i in range(0, n):
    for j in range(0, n): 
        if i<(n-1):
            y_label=label[0]
            y_labels.append(y_label)
        else:
            y_label=label[0]
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
    num_classes = 6
    epochs = 200
    
    ## Convert class vectors to binary class matrices.
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    #print (y_train)
    #print (y_test)
    
    #### Scale 0-255 bands range into float 0-1 ####
    x_train = x_train.astype('float32')
    
    x_test = x_test.astype('float32')
    
    np.asarray(x_train)
    np.asarray(x_test)
    np.asarray(y_train)
    np.asarray(y_test)
  
    from tempfile import TemporaryFile
    training_data = TemporaryFile()
    np.savez_compressed('training_data', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
