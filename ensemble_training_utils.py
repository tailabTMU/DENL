

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import time
from scipy.stats import mode
import glob
from setup_cifar import load_batch
from tensorflow.keras import Model
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix


def extract_temp(teacher_id,
                 path, 
                 server=True ):
    if server== True:
        description_path = f'/teacher_{teacher_id}/model_description.txt'
    else:
        description_path = f'\\teacher_{teacher_id}\\model_description.txt'
    with open(path+description_path) as f:
        lines = f.readlines()
        print('lines',lines)
    a = list(lines[0])
    for i in range(0,len(a)):
        if a[i]==':':
            start_index = i+1
        if a[i]=='\n':
            finish_index = i
    tem =a[start_index:finish_index]
    temp = int(''.join(tem))
    return temp
    

def tfn(train_temp):
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
    return fn


def load_members(folder_path, subfolder_path_individuals, temp_limit, server=True):
    #Finding the path to all models
    print("folder_path. subfolder_path",folder_path,subfolder_path_individuals)
    if server==True:
        sub_folder_names = glob.glob('{}/{}*'.format(folder_path, subfolder_path_individuals))
    else:
        sub_folder_names = glob.glob('{}\\{}*'.format(folder_path, subfolder_path_individuals))
    #counting the trained teachers of the experiment
    print('sub folder names:',sub_folder_names)
    number_of_models = len(sub_folder_names)
    members=[]
    print('number of models',number_of_models)
    for i in range(number_of_models):
    #extracting the temperature of each teacher using the extract_temp function
        t = extract_temp(i, path=folder_path,server=server)
        print('temperature:',t)
        if t<temp_limit:
            model_path = sub_folder_names[i]
            model = load_model(model_path, custom_objects = {'fn':tfn(t)})
   #test_predicts = model.predict(test_data)
            members.append(model)
    print(len(members),'len(members)')
    return members

def tf_pearson_simple(x, y):
    '''
    This function will be used to compute the Pearson corrolation for each class over each batch of samples'
    '''
    #print('len(x)',len(x))
    #print('len(y)',len(y))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    mx = tf.math.reduce_mean(input_tensor=x)
    my = tf.math.reduce_mean(input_tensor=y)
    #print('mx,my:',mx,my)
    xm, ym = x-mx, y-my
    #print('xm,ym:',xm,ym)
    C = tf.multiply(xm,ym)
    #print('multiplication',C)
    r_num = tf.math.reduce_mean(input_tensor=C)
    #print('r_num:',r_num)
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    r_den += 1e-10
    return  (r_num / r_den)

def rll(O):
    n=len(O)
    #print('n:',n)
    LL=0
    m=O[0].shape[1]
    counter=0
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if j>i:
                    coef_value=tf_pearson_simple(O[i][:,k],O[j][:,k])
                    #print(coef_value)
                    #print(O[i].shape,O[j].shape)
                    #print(O[i])
                    LL+=coef_value
                    #print(LL)
                    counter+=1
        RLL=2.0*LL/(m*n*(n-1))
    #print(counter)
    #print((m*n*(n-1))/2)
    return RLL

def rtl_symmetric(O,Y,rowvar=False):
    m = O[0].shape[1]
    #print('m:',m)
    n=len(O)
    #print('n:',n)
    TL=0
    for k in range(m):
        for i in range(n):
            coef_value=tf_pearson_simple(O[i][:,k],Y[i][:,k])
            TL+=coef_value
            #print(coef_value)
    RTL=(tf.cast(TL,tf.float32))/(n*m)
    return RTL

# def rte_symmetric(O,Y,rowvar=False):
#     m = O[0].shape[1]
#     #print('m:',m)
#     n=len(O)
#     #print('n:',n)
#     TL=0
#     ens_labels=[]
#     ens_labels = [np.argmax(O[i],axis=0) for i in range(n)]
#     for one in ens_labels:
#         ens_labels_zero = np.zeros((one.size,one.max()+1))
#         ens_labels_zero[np.arange(one.size),one]=1
#         ens_labels.append(ens_labels_zero)
#     for k in range(m):     
#         coef_value=tf_pearson_simple(O[i][:,k],Y[i][:,k])
#         TL+=coef_value
#         #print(coef_value)
#     RTL=(tf.cast(TL,tf.float32))/m
#     return RTL

def rtl_symmetric_new(O,Y,rowvar=False):
    #To Do : convert the for to numpy array form
    m = O[0].shape[1]
    #print('m:',m)
    n=len(O)
    #print('n:',n)
    TL=0
    for i in range(n):       
        hard_labels = np.zeros(O[i].shape)
        indices = np.argmax(O[i],axis=1)
        hard_labels[range(hard_labels.shape[0]),indices]=1
        for k in range(m):  
            coef_value=tf_pearson_simple(hard_labels[:,k],Y[i][:,k])
            TL+=coef_value
            #print(coef_value)
    RTL=(tf.cast(TL,tf.float32))/(n*m)
    return RTL

def main_acc(o,Y,return_correct_indices=False):
    '''
    This function is for evaluating the accuracy of the ensemble by comparing the ensemble model predictions with the labels
    '''
    max_index_all = []
    for i in range(len(o)):
        # returning the index of the class with the highest probability
        max_index = np.argmax(o[i], axis=1)
        # Collecting the predicted class of all models
        max_index_all.append(max_index)
    max_index_array = np.array(max_index_all)
    #max_index_array is of the shape (number_of_models,number_of_testpoint)
    #print(max_index_array.shape)
    # getting the mode in order to see the result of voting
    array_mode, array_count = mode(max_index_array , axis=0)
    #Converting the one_hot_encoding format of the test_labels to class labels
    labels = np.argmax(Y, axis=1)
    # Computing the accuracy
    accu = np.sum(array_mode == labels)/len(labels)
    if return_correct_indices==False:
        return accu
    else:
        correct_indices = array_mode == labels
        #tmp = max_index_array[:,correct_indices.reshape(-1,)][:,:4]
        #print('tmp:',tmp)
        return accu , correct_indices.reshape(-1,)
        

def ensemble_confusion_matrix(o,Y):
    '''
    This function is for generating the confusion matrix of the ensemble by comparing the ensemble model predictions with the labels
    '''
    max_index_all = []
    for i in range(len(o)):
        # returning the index of the class with the highest probability
        max_index = np.argmax(o[i], axis=1)
        # Collecting the predicted class of all models
        max_index_all.append(max_index)
    max_index_array = np.array(max_index_all)
    #max_index_array is of the shape (number_of_models,number_of_testpoint)
    #print(max_index_array.shape)
    # getting the mode in order to see the result of voting
    array_mode, array_count = mode(max_index_array , axis=0)
    array_mode = array_mode.reshape(array_mode.shape[1],)
    #Converting the one_hot_encoding format of the test_labels to class labels
    labels = np.argmax(Y, axis=1)
    # Computing the ensemble confusion matrix
    print('shape of labels', labels.shape , 'shape of array mode' , array_mode.shape)
    confusion = confusion_matrix(y_true=labels,y_pred=array_mode)
    
    return confusion

def loss_function_symmetric(y_pred,y_true,lambdaa,pang=False):
    #print('len O:'+str(len(O)))
    
    rll_value = rll(y_pred)
    if pang:
        n_members = len(y_pred)
        temp_list = [(i+1)*10 for i in range(n_members)]
        cross_loss = 0
        for j,temp in enumerate(temp_list):
            cross_entropy = tfn(temp)
            cross_loss = cross_loss + cross_entropy(y_true[j],(y_pred[j]+1e-7))
            # print('y pred shape:',y_pred[j].shape,'y true shape',y_true[j].shape)
            # print('temp:',temp)            
        cross_loss = cross_loss/n_members
        print('cross loss:',cross_loss)        
        loss = (lambdaa)*rll_value + cross_loss 
        return loss , cross_loss ,rll_value
    else:
        rtl_value = rtl_symmetric(y_pred,y_true)
        loss=-(rtl_value-(lambdaa)*rll_value)
        #print('Shape of y'+str(Y.shape))
        #print('Shape of ensemble output'+str(O[-1].shape))
        
        return loss , rtl_value ,rll_value



def loss(model, x, y, training, lambdaa, n_members,pang= False):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  x_list = [x for i in range(n_members)]
  y_list = [y for i in range(n_members)]
  y_pred = model(x_list , training=training)
  #print('y_pred',y_pred)
  l , rtl_value , rll_value = loss_function_symmetric(y_true=y_list,
                              y_pred=y_pred,
                              lambdaa=lambdaa,
                              pang=pang)
  return l , rtl_value , rll_value

def grad(model, inputs, targets, lambdaa, n_members, training=True, pang= False):
  with tf.GradientTape() as tape:
    #print('len(input):',len(inputs))
    #print('len(targets):',len(targets))
    #print('shape(input):', inputs.shape)
    #print('shape(targets):',targets.shape)
    output = loss(model,
                  inputs,
                  targets,
                  training=training,
                  lambdaa=lambdaa,
                  n_members=n_members,
                  pang=pang)
    loss_value = output[0]
    #print('loss:', loss_value.numpy())
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def define_stacked_model_simple(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = True
            # make not trainable
            #if "dense" in layer.name:
            #    layer.trainable = True
            #else:
            #    layer.trainable = False         
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
            print(layer._name)
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    #all_outputs=Lambda(lambda x: x,name='member_output')(ensemble_outputs)
    #merge = concatenate(ensemble_outputs)
    #hidden = Dense(20, activation='relu')(merge)
    #final_output = Dense(10, activation='softmax',name='final_output')(hidden)
    #all_outputs.append(output)
    #output_all = concatenate(output)
    model = Model(inputs=ensemble_visible, outputs=ensemble_outputs)
    print('model summary:',model.summary())
    #loss_func   =     {'member_output':pearson_loss , 'final_output':'categorical_crossentropy'}
    #loss_weights =     {'member_output':0.5, 'final_output':0.5}
    # plot graph of ensemble
    #plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    #model.compile(loss=loss_function_symmetric , optimizer='adam', metrics=['accuracy'])
    return model


def compute_test_metrics(model,
                         test_data,
                         test_labels,
                         lambdaa,
                         n_members, 
                         batch_size,
                         pang=False):
    
    num_batches = (len(test_data)//batch_size)+1
    epoch_loss_test = tf.keras.metrics.Mean()
    epoch_accuracy_test = tf.keras.metrics.Mean()
    epoch_rtl_test = tf.keras.metrics.Mean()
    epoch_rll_test = tf.keras.metrics.Mean()
    for a in range(num_batches):
        test_data_batch = test_data[a*batch_size:(a+1)*batch_size]
        test_labels_batch = test_labels[a*batch_size:(a+1)*batch_size]
        output = loss(model,
                      test_data_batch,
                      test_labels_batch,
                      lambdaa=lambdaa,
                      n_members=n_members,
                      training=False,
                      pang=pang)
        #print('Output for test metrics:', output)
        loss_value_test = output[0]
        epoch_rtl_test.update_state(output[1])
        epoch_rll_test.update_state(output[2])
        
        epoch_loss_test.update_state(loss_value_test)
        test_xinput = [test_data_batch for i in range(n_members)]
        epoch_accuracy_test.update_state(main_acc(o= model(test_xinput,
                                                           training=False),
                                                  Y=test_labels_batch))
        
    return (epoch_loss_test.result(), epoch_accuracy_test.result() ,
            epoch_rtl_test.result() , epoch_rll_test.result())
  
    
def ensemble_training(train_data_all, 
                      train_labels_all,
                      ensemble_train_batch_size,
                      num_epochs_ensemble_training,
                      lambdaa,
                      optimizer,
                      model,
                      folder_path,
                      subfolder_path_ensemble_learning,
                      logger_path,
                      n_members,
                      test_data,
                      test_labels,
                      ensemble_test_batch_size,
                      server=True,
                      pang = False
                      ):
    
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    test_rtl_results = []
    test_rll_results = []
    #batch_size = batch_size
    num_steps = (len(train_data_all)//ensemble_train_batch_size)+1
    indices = [a for a in range(len(train_data_all))] 
    for epoch in range(num_epochs_ensemble_training):
      print('Epoch numnber:',epoch,'started')
      start_time = time.time()
      random.shuffle(indices)
      epoch_loss_avg = tf.keras.metrics.Mean()
      #epoch_accuracy =http://localhost:8888/notebooks/Documents/GitHub/Yuting_code_modifications/ensemble_loss_grad.ipynb# tf.keras.metrics.SparseCategoricalAccuracy()
      epoch_accuracy = tf.keras.metrics.Mean()
    
      # Training loop - using batches of 32
      #for x, y in ds_train_batch:
      #for x,y in zip(train_data_all,train_labels_all):
      for step in range(num_steps):
        # To Do : use data loader to have larger batches 
        x = train_data_all[indices[step*ensemble_train_batch_size:(step+1)*ensemble_train_batch_size]]
        y = train_labels_all[indices[step*ensemble_train_batch_size:(step+1)*ensemble_train_batch_size]]
        #x = tf.expand_dims(x , axis=0)
        #y = tf.expand_dims(y , axis=0)
        #print('x:',x,'y:',y)
        # Optimize the model
        loss_value, grads = grad(model, x, y, n_members=n_members,
                                 lambdaa=lambdaa, pang=pang)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #print('len(grads)',len(grads),'len(model.trainable_variables)',len(model.trainable_variables))
    
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        model_xinput = [x for i in range(n_members)]
        #epoch_accuracy.update_state(y, model(model_xinput, training=True))
        epoch_accuracy.update_state(main_acc(o= model(model_xinput, training=False),Y=y))
      
      
      # End epoch
      train_loss_results.append(epoch_loss_avg.result())
      train_accuracy_results.append(epoch_accuracy.result())
      
      test_output = compute_test_metrics(model,
                             test_data,
                             test_labels,
                             n_members=n_members, 
                             batch_size= ensemble_test_batch_size,
                             lambdaa=lambdaa,
                             pang=pang)
      epoch_loss_test = test_output[0]
      epoch_accuracy_test = test_output[1]
      epoch_rtl_test = test_output[2]
      epoch_rll_test = test_output[3]
      
      test_rtl_results.append(epoch_rtl_test)
      test_rll_results.append(epoch_rll_test)
      test_loss_results.append(epoch_loss_test)
      test_accuracy_results.append(epoch_accuracy_test)
        
      if epoch%1 == 0:
          if server== True:
              model.save(folder_path+ subfolder_path_ensemble_learning+f'/epoch_{epoch}')
          else:
              model.save(folder_path+ subfolder_path_ensemble_learning+f'\\epoch_{epoch}')
                     
          
      print("""Epoch {:03d}: Training Loss: {:.7f}, Training Accuracy: {:.7%},
            Test Loss:{:.7f}, Test Accuracy:{:.7%},
            Test_rtl:{:.7%} , Test_rll:{:.7%}""".format(epoch,
                                                             epoch_loss_avg.result(),
                                                             epoch_accuracy.result(),
                                                             epoch_loss_test,
                                                             epoch_accuracy_test,
                                                             epoch_rtl_test,
                                                             epoch_rll_test))
      end_time = time.time()
      print('Time of the epoch:',str(end_time-start_time))
      my_dict = {'epoch':epoch , 'training loss':epoch_loss_avg.result().numpy(),
                 'training accuracy':epoch_accuracy.result().numpy(),
                 'test loss':epoch_loss_test.numpy(),
                 'test accuracy': epoch_accuracy_test.numpy(),
                 'test rtl':epoch_rtl_test.numpy(),
                 'test rll':epoch_rll_test.numpy()}
      writing_mode = "w" if epoch==0 else "a"
      with open(folder_path+subfolder_path_ensemble_learning+logger_path,
                writing_mode) as f:
            if epoch==0:
                f.write("%s,%s,%s,%s,%s,%s,%s\n"%("epoch",
                                            "training loss",
                                            "training accuracy",
                                            "test loss",
                                            "test accuracy",
                                            "test rtl",
                                            "test rll")) 
            f.write("%s,%s,%s,%s,%s,%s,%s\n"%(my_dict["epoch"],
                                        my_dict["training loss"],
                                        my_dict["training accuracy"],
                                        my_dict["test loss"],
                                        my_dict["test accuracy"],
                                        my_dict["test rtl"],
                                        my_dict["test rll"]))
    return 


def plotting(folder_path,subfolder_path_ensemble_learning,logger_path):
    
    logger_data = pd.read_csv(folder_path+subfolder_path_ensemble_learning+logger_path)
    plt.figure(num=1, figsize=(8,8))
    plt.subplot(2,2,1)
    plt.plot(logger_data['training loss'])
    plt.title('training loss')
    plt.grid(True)
    plt.subplot(2,2,2)
    plt.plot(logger_data['training accuracy'])
    plt.grid(True)
    plt.ylim((0,1.1))
    plt.title('training accuracy')
    plt.subplot(2,2,3)
    plt.plot(logger_data['test loss'])
    plt.grid(True)
    plt.title('test loss')
    plt.subplot(2,2,4)
    plt.plot(logger_data['test accuracy'])
    plt.grid(True)
    plt.ylim((0,1))
    plt.title('test accuracy')
    plt.savefig(os.path.join(folder_path+subfolder_path_ensemble_learning,'plot.png'))
    plt.show()

def plot_training_accuracy_curves(folder_path,
                                  num_teachers):
       
    for i in range(num_teachers):
        file_path1 = os.path.join(folder_path,'teacher_'+str(i))
        print('file_path1:' ,file_path1)
        file_path = os.path.join(file_path1,'logger.csv')
        print('file_path:',file_path)
        data = pd.read_csv(file_path)
        print(data.head(2))
        plt.figure(0, figsize=(8, 8))
        plt.plot(data.loc[:, "accuracy"],
                 label= 'model'+str(i)+ ' temperature: '+ str((i+1)*10))
        plt.title("training accuracy" , fontsize=18)
        plt.xlabel("number of epochs", fontsize=18)
        plt.ylabel("classification accuracy", fontsize=18)
        plt.grid(True)
        lgd=plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.savefig(os.path.join(folder_path,"training_accuracy_curve_for_model.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.figure(1, figsize=(8, 8))
        plt.plot(data.loc[:, "val_accuracy"], 
                 label= 'model'+str(i)+ ' temperature: '+ str((i+1)*10))
        plt.title("validation accuracy", fontsize=18)
        plt.xlabel("number of epochs", fontsize=18)
        plt.ylabel("classification accuracy", fontsize=18)
        plt.grid(True)
        lgd=plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(os.path.join(folder_path,"validation_accuracy_curve_for_model.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plt.close()
    plt.show()


   
def break_stacked_model(stacked_model,
                        n_members,
                        train_temp,
                        input_shape=(32, 32, 3)):
    
    members_layers = [[] for i in range(n_members)]
    for i in range(n_members):
        members_layers[i].append(InputLayer(input_shape=input_shape))
    for layer in stacked_model.layers:
        if 'ensemble' in layer._name:
            #To Do: Write the next line for ensemble members more than 10
            member_index = int(layer._name.split("_")[1])
            #print(member_index)
            members_layers[member_index-1].append(layer)
    members = [Sequential(members_layers[i]) for i in range(n_members)]
    sgd = SGD(learning_rate=0.01,  momentum=0.9, decay=1e-6, nesterov=True)
    for i,model in enumerate(members):
        model.compile(loss=tfn(train_temp[i]),
                  optimizer=sgd,
                  metrics=['accuracy'])
    return members
    


if __name__ == "__main__":
        
    server = False
    dataset = 'CIFAR' #'MNIST'
    architecture = 'similar' #'diverse'
    num_teachers = 5
    num_epochs_individual_training = 75
    num_epochs_ensemble_training = 75
    learning_rate = 0.1
    lambdaa = 0.9
    temp_limit = 55
    batch_size = 5
    
    if server == True:
        folder_path = f'/home/grad/m3yazdan/Github/Results/{dataset}_{architecture}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers'
        subfolder_path_individuals = 'teacher'
        subfolder_path_ensemble_learning = '/{num_epochs_ensemble_training}ensemble_epochs_lambda_{lambdaa }_learningrate_{learning_rate}/' 
        logger_path = '/logger.csv'
    
    else:
        folder_path = f'C:\\Users\\miyaz\\Documents\\GitHub\\Results\\{dataset}_{architecture}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers'    
        subfolder_path_individuals = '\\teacher'
        subfolder_path_ensemble_learning = '\\{num_epochs_ensemble_training}ensemble_epochs_lambda_{lambdaa }_learningrate_{learning_rate}' 
        logger_path = '\\logger.csv'



    train_data = []
    train_labels = []
    if server==True:
        data_path = '/home/grad/m3yazdan/Github/Yuting_code_modifications/cifar10_batches_bin/'
    else:
        data_path = 'C:\\Users\\miyaz\\Documents\\GitHub\\Yuting_code_modifications\\cifar10_batches_bin\\'
    for j in range(1,6):
        data,labels = load_batch(data_path+'data_batch_'+str(j)+'.bin')
        train_data.append(data)
        train_labels.append(labels)
    
    train_data_all = np.concatenate((train_data[0],train_data[1],train_data[2],train_data[3],train_data[4]),axis= 0)
    train_labels_all = np.concatenate((train_labels[0],train_labels[1],train_labels[2],train_labels[3],train_labels[4]),axis= 0)
    train_data_all = train_data_all[:1000]
    train_labels_all = train_labels_all[:1000]
    print(train_data_all.shape)
    print(train_labels_all.shape)
    
    test_data,test_labels = load_batch(data_path+'test_batch.bin')
    test_labels = test_labels.astype(int)
    test_data = test_data[:1000]
    test_labels = test_labels[:1000]
    
    members=load_members(folder_path,
                         subfolder_path_individuals,
                         temp_limit=temp_limit) 
    
    n_members = len(members)
    print(n_members)
      
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model=define_stacked_model_simple(members)
    
    ensemble_training(train_data_all, 
                      batch_size,
                      num_epochs_ensemble_training,
                      lambdaa,
                      optimizer,
                      model,
                      folder_path,
                      subfolder_path_ensemble_learning,
                      logger_path,
                      n_members=n_members,
                      test_data=test_data,
                      test_labels=test_labels
                      )
    
    
    plotting(folder_path,subfolder_path_ensemble_learning,logger_path)
    
    
    
