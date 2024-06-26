
# This code is originated from the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini, David Wagner
import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

def tfn(train_temp):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
    return fn

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self,fold):
        # fold is an integer between 0 , 9 for random selection of validation data 
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")
        self.test_data_correct, self.test_labels_correct = self.test_data, self.test_labels
        VALIDATION_SIZE = 5000
        index_set = list(np.arange(len(train_data)))
        validation_index_start = fold*VALIDATION_SIZE
        validation_index_end = (fold+1)*VALIDATION_SIZE
        validation_set = list(np.arange(validation_index_start,
                                        validation_index_end))
        training_set = [i for i in index_set if i not in validation_set]
        
        self.validation_data = train_data[validation_set]
        self.validation_labels = train_labels[validation_set]
        self.train_data = train_data[training_set]
        self.train_labels = train_labels[training_set]
        
        
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]
        

class CIFARModel:
    def __init__(self, params, temp, restore=None, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        if restore is None:
            model = Sequential()
            
            model.add(Conv2D(params[0], (3, 3),
                                    input_shape=(32, 32, 3)))
            model.add(Activation('relu'))
            model.add(Conv2D(params[1], (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
            model.add(Conv2D(params[2], (3, 3)))
            model.add(Activation('relu'))
            if len(params)==5:
                dense_index=3
            else:
                model.add(Conv2D(params[3], (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            if len(params)==6:
                dense_index=4
            if len(params)==7:
                dense_index=5
                model.add(Conv2D(params[dense_index-1],(3, 3)))
                model.add(Activation('relu'))
            if len(params)==8:
                dense_index=6
                model.add(Conv2D(params[dense_index-2],(3, 3)))
                model.add(Activation('relu'))
                model.add(Conv2D(params[dense_index-1],(3, 3)))
                model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(params[dense_index]))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(params[dense_index+1]))
            model.add(Activation('relu'))
            model.add(Dense(10))
        else:

            model = load_model(restore, custom_objects = {'fn':tfn(temp)})

        self.model = model

    def predict(self, data):
        return self.model(data)

## modified model definition
class CIFARDPModel:
    def __init__(self, params, temp, restore=None, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        
        if restore is None:
            
            model = Sequential()
            model.add(Conv2D(params[0], (3, 3),
                                    input_shape=(32, 32, 3)))
            model.add(Activation('relu'))
            model.add(Conv2D(params[1], (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
            model.add(Conv2D(params[2], (3, 3)))
            model.add(Activation('relu'))
            model.add(Conv2D(params[3], (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if len(params)==6:
                dense_index=4
            if len(params)==7:
                dense_index=5
                model.add(Conv2D(params[dense_index-1],(3, 3)))
            if len(params)==8:
                dense_index=6
                model.add(Conv2D(params[dense_index-2],(3,3)))
                model.add(Conv2D(params[dense_index-1],(3, 3)))
            model.add(Flatten())
            model.add(Dense(params[dense_index]))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(params[dense_index+1]))
            model.add(Activation('relu'))
            model.add(Dense(10))
        else:

            model = load_model(restore, custom_objects = {'fn':tfn(temp)})

        self.model = model


    def predict(self, data):
        shape = []
        if isinstance(data.shape[0], int):
            shape = list(data.shape)
        elif not data.shape[0] == None:
            shape = [data.shape[i] for i in range(0, data.shape.ndims)]
        else:
            shape = [data.shape[i] for i in range(1, data.shape.ndims)]
        loc = np.zeros(shape)
        return self.model(data+np.random.normal(loc, scale = 0.03))
        
    
