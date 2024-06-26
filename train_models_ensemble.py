# This code is originated from the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini, David Wagner


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import tensorflow as tf
from setup_mnist import MNIST, MNISTModel
# from setup_cifar import CIFAR
import os

np.random.seed(seed=0)

def train(train_data,
          train_labels,
          validation_data,
          validation_labels,    
          file_name,
          params,
          num_epochs=50,
          batch_size=128,
          train_temp=1,
          init=None):
    """
    Standard neural network training procedure.
    """
    
    model = Sequential()

    print(train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=train_data.shape[1:]))
    print('data input shape:', train_data.shape[1:])
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
        model.add(Conv2D(params[dense_index-1],(3,3)))
    model.add(Flatten())
    model.add(Dense(params[dense_index]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[dense_index+1]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #, decay=1e-6
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    
    callbacks = [ModelCheckpoint(filepath=file_name, save_best_only=True, metric="val_accuracy"),
                 CSVLogger(filename=file_name+"/logger.csv")]
    
    model.fit(train_data,
              train_labels,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              epochs=num_epochs,
              callbacks=callbacks,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return True


## train all teacher models based on original train function by Nicolas Carlini
## use partitioning of training data for MNIST, no partitioning for CIFAR
def train_teachers_all(dataset,
                       params_list,
                       arrTemp,
                       train_fileprefix,
                       arrInit=None,
                       num_epochs=50,
                       id_start=0,
                       bool_partition=True,
                       bool_bagging=False,
                       partial_data_usage = False):
  members=[]  
  nb_teachers = len(arrTemp)
  for teacher_id in range(id_start, nb_teachers):
    # Retrieve subset of data for this teacher
    
    params = params_list[teacher_id]
    #params = params_list[teacher_id]
    print('**params:', params)
    
    temp = arrTemp[teacher_id]
    #temp=1
    print('*Temperature:', temp)
    #tempcounter+=1
    if bool_partition:
      train_data, train_labels = partition_dataset(dataset.train_data,
                                            dataset.train_labels,
                                            nb_teachers,
                                            teacher_id)
    elif bool_bagging:
      train_data, train_labels = bagging_dataset(dataset.train_data,
                                            dataset.train_labels,
                                            nb_teachers,
                                            teacher_id)
        
    else:
      train_data, train_labels = dataset.train_data, dataset.train_labels
    
    if partial_data_usage:
        train_data, train_labels = train_data[0:1000],train_labels[0:1000]
        

    print("Length of training data: " + str(len(train_labels)))

    # Define teacher checkpoint filename and full path
    filename = os.path.join(train_fileprefix,'teacher_' + str(teacher_id))
    #filename = 'teachers_' + str(teacher_id)

    
    init = None
    if arrInit != None:
      init = arrInit[teacher_id]
    
    train(train_data,
          train_labels,
          dataset.validation_data,
          dataset.validation_labels,
          filename,params,
          num_epochs=num_epochs,
          train_temp=temp,
          init=init)
    
    
    model_description = open(filename + '/model_description.txt','w+')
    model_description.write('*Temperature:'+ str(temp))
    model_description.write('\n **params:'+ str(params))
    model_description.close()
    
  return True

## partition function for training data
def partition_dataset(data, labels, nb_teachers, teacher_id):

  # Sanity check
  assert len(data) == len(labels)
  assert int(teacher_id) < int(nb_teachers)

  # This will floor the possible number of batches
  batch_len = int(len(data) / nb_teachers)

  # Compute start, end indices of partition
  start = teacher_id * batch_len
  end = (teacher_id+1) * batch_len

  # Slice partition off
  partition_data = data[start:end]
  partition_labels = labels[start:end]

  return partition_data, partition_labels

def bagging_dataset(data, labels, nb_teachers, teacher_id):
    
    # Sanity check
    assert len(data) == len(labels)
    assert int(teacher_id) < int(nb_teachers)
    
    # This will floor the possible number of batches
    batch_len = int(len(data) / nb_teachers)
    
    ran = np.random.choice(range(0,len(data)),
                           size=batch_len,
                           replace=True,
                           p=None)
    
    bagging_data = data[ran]
    bagging_labels = labels[ran]
    
    return bagging_data, bagging_labels
    
    
    
    
def check_accuracy(model, test_data, test_labels, image_size, num_channels, batch_size=1):
  with tf.Session() as sess:
    x = tf.placeholder(tf.float32, (None, image_size, image_size, num_channels))
    y = model.predict(x)
    r = []
    for i in range(0,len(test_data),batch_size):
        pred = sess.run(y, {x: test_data[i:i+batch_size]})
        r.append(np.argmax(pred,1) == np.argmax(test_labels[i:i+batch_size],1))
        print(np.mean(r))
    return np.mean(r)


