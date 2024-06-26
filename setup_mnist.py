

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model


def tfn(train_temp):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
    return fn

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self,fold):
        '''
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
        '''
        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        self.test_data_correct, self.test_labels_correct = self.test_data, self.test_labels
        
        VALIDATION_SIZE = 5000
                
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

class MNISTDP:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

        
class MNISTModel:
    def __init__(self, params, temp, restore=None, session=None ):
        # restore points to the address of saved models 
        #PArams points to the architecture of the model
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        if restore is None:
            model = Sequential()
    
            model.add(Conv2D(params[0], (3, 3),
                                    input_shape=(28, 28, 1)))
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
                model.add(Activation('relu'))
            if len(params)==8:
                dense_index=6
                model.add(Conv2D(params[dense_index-2],(3,3)))
                model.add(Activation('relu'))
                model.add(Conv2D(params[dense_index-1],(3,3)))
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


class MNISTDPModel:
    def __init__(self, params, temp, restore=None, session=None ):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        if restore is None:
            model = Sequential()
    
            model.add(Conv2D(params[0], (3, 3),
                                    input_shape=(28, 28, 1)))
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
                model.add(Activation('relu'))
            if len(params)==8:
                dense_index=6
                model.add(Conv2D(params[dense_index-2],(3,3)))
                model.add(Activation('relu'))
                model.add(Conv2D(params[dense_index-1],(3,3)))
                model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(params[dense_index]))
            model.add(Activation('relu'))
            # Yuting in this class removed line 161 (the dropout layer)
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
        #loc = np.zeros(data.shape)
        return self.model(data+np.random.laplace(loc, scale = 0.5))

    def model_predict(self,data):
        
        shape = []
        if isinstance(data.shape[0], int):
            shape = list(data.shape)
        elif not data.shape[0] == None:
            shape = [data.shape[i] for i in range(0, data.shape.ndims)]
        else:
            shape = [data.shape[i] for i in range(1, data.shape.ndims)]
        loc = np.zeros(shape)
    
        #loc = np.zeros(data.shape)
        return self.model.predict(data+np.random.normal(loc, scale = 0.5))

