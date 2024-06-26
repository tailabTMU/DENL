
from train_models_ensemble import train_teachers_all
from setup_mnist import MNIST
from setup_cifar import CIFAR
import time
import tensorflow as tf
import numpy as np
import os
import random
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
from setup_cifar import load_batch
from ensemble_training_utils import load_members , ensemble_confusion_matrix
from ensemble_training_utils import define_stacked_model_simple
from ensemble_training_utils import ensemble_training
from ensemble_training_utils import plotting, plot_training_accuracy_curves
from ensemble_training_utils import break_stacked_model
from ensemble_training_utils import rtl_symmetric ,rtl_symmetric_new , rll ,main_acc
from test_ensemble_func import test_ensemble_noisylogit_superimposed_opt, plotCIFAR, plotMNIST, makeDir, test_ensemble_noisylogit_superimposed_all_byinput
from setup_cifar import CIFAR, CIFARModel, CIFARDPModel
from setup_mnist import MNIST, MNISTModel, MNISTDPModel
from process_results import individual_statistics, single_process , process_all_results,get_min_max_details, extract_stats, get_transfer_stats #, get_stats_by_bucket_arrfiles
import glob
from sklearn.metrics import confusion_matrix


 
####### Flags: You can change the flags  ######### 
# The training data
dataset = 'CIFAR' #'CIFAR' # 'MNIST'
# The architecture of the networks
architecture =  'diverse' #'single'  #  'similar'
# If we add noise to the inputs
noisy_logit = False
# Activate the cross-validation
cross_validation = True
# Number of the networks in an ensemble
num_teachers = 10
# lambdaa is the lambda parameter required for the loss function of phase 2
lambdaa = 0.4
num_epochs_ensemble_training = 150
learning_rate = 0.01  #phase2
temp_limit = num_teachers*10 +1
if architecture == 'single':
    num_teachers = 1
    temp_limit = 2
ensemble_train_batch_size = 512 #phase2
ensemble_test_batch_size =  1024 #phase2
number_of_attacked_samples = 128
number_of_attacked_samples_sup = 256
nstart_attack = 0
attack_batch_size = 64
phase1_training = True
phase2_training = True
phase3_breaking = True
phase4_single_attack = True
phase4_sup_attack = False
Attack_phase1_models =  True
#treshhold=1
server = False
architecture_num = 0.2
pang = False
phase5_process_single_attack = True
phase5_process_sup_attack = False
load_data_model = True
attacked_single_statistics =True
nSup = 2 # 3
partial_data_usage = False
bool_partition = False
bool_bagging = False
random_architecture = False
if dataset == 'MNIST':
    if pang == True:
        num_epochs_individual_training =1
    else:
        num_epochs_individual_training = 250
    bool_partition = True
    attack_max_iterations=500
    attack_learning_rate=0.01
    break_input_shape=(28, 28, 1)
    
if dataset == 'CIFAR':
    if pang ==True:
        num_epochs_individual_training = 1
    else:
        num_epochs_individual_training = 150
    bool_partition = False
    attack_max_iterations=50
    attack_learning_rate=0.01
    break_input_shape=(32, 32, 3)
    
num_single_adv_examples = num_teachers*9*number_of_attacked_samples

#if phase1_training ==False:
#   cross_validation = False
    
confusion_acc_list = []

noisy_confusion_acc_list = []
if cross_validation == True:
    fold_index_set = range(9,-1,-2) #For having 5 folds of 9-7-5-3-1    
    #fold_index_set = random.sample(range(0,10), num_folds)
else:
    fold_index_set =[1]
 
# %%
# Architecture creation section    
def random_architecture_creator(base_params,seed):
    np.random.seed(seed)
    #last two layers are dense layers.
    num_convolution_layers_base = len(base_params)-2
    num_convolution_layers = np.random.randint(low=num_convolution_layers_base-1,
                                               high=num_convolution_layers_base+2)
    base_power = 6            
    size_power = np.random.randint(low=base_power-1,
                                   high=base_power+2,
                                   size=num_convolution_layers)
    architecture = [2**i for i in size_power]+base_params[-2:]
    return architecture

if dataset=='CIFAR':
    
    base_params = [64, 64, 64, 64, 256, 256]
    params = []    
    
    if architecture == 'similar':
        for i in range(num_teachers):
            params.append(base_params)
    elif architecture == 'diverse':
        if random_architecture == True:
            for i in range(num_teachers):
                params.append(random_architecture_creator(base_params,i))    
        
        else:
            # Architecture 0.2
            if architecture_num == 0.2:
                params_0 = [64, 64, 64, 64, 256, 256]
                params_1 = [32, 32, 64, 64 , 64, 128, 256, 256]
                params_2 = [64, 64, 128, 128, 128, 128, 256, 256]
                params_3 = [64, 64, 128, 128, 128, 256, 256]
                params_4 = [128 ,128 , 128, 256, 256, 256, 256]
                params_5 = [64, 64, 256, 256, 256, 256]
                params_6 = [64, 64, 128, 128 , 256, 256, 256, 256]
                params_7 = [128, 128, 128, 128, 128, 128, 256, 256]
                params_8 = [128, 128, 128, 512, 512, 256, 256]
                params_9 = [128,128 ,128 , 128, 256, 256, 512, 512]
                params = [eval('params_'+str(i)) for i in range(num_teachers)]
                
    elif architecture == 'single':
        params.append(base_params)
        
if dataset=='MNIST':
    
    base_params = [32, 32, 64, 64, 200, 200]
    params = []    
    
    if architecture == 'similar':
        for i in range(num_teachers):
            params.append(base_params)
    elif architecture == 'diverse':
        if random_architecture == True:
            for i in range(num_teachers):
                params.append(random_architecture_creator(base_params,i))    
        
        else:
            # Architecture 0.2
            if architecture_num == 0.2:
                params_0 = [32, 32, 64, 64, 200, 200]
                params_1 = [16, 16, 64, 64 , 100, 100]
                params_2 = [32, 32, 32, 32 ,  200, 200]
                params_3 = [32, 32, 32, 32 , 32, 400, 400]
                params_4 = [32, 32, 32, 64 , 64, 400, 400]
                params_5 = [64, 64 , 64, 64, 400, 400]
                params_6 = [64, 64 , 64, 64 , 64, 400, 400]
                params_7 = [64, 64 , 64, 64 , 64, 200, 200]
                params_8 = [64, 64, 64 , 128, 128 , 200, 200]
                params_9 = [32, 32, 32, 32 , 64 , 200, 200]
                params = [eval('params_'+str(i)) for i in range(num_teachers)]
                
    elif architecture == 'single':
        params.append(base_params)
        
        
print('created params:', params)

# %%
# Creating plotting and noise class

if dataset == 'MNIST':
    plot_img = plotMNIST
    noise_scale = 0.5
    if noisy_logit == False:
        dataset_model = MNISTModel
    else:
        dataset_model = MNISTDPModel
elif dataset == 'CIFAR':
    noise_scale = 0.03
    plot_img = plotCIFAR
    if noisy_logit == False:
        dataset_model = CIFARModel 
    else:
        dataset_model = CIFARDPModel
        
arr_temp = [i*10 for i in range(1,(temp_limit//10)+1,1)]
if architecture == 'single':
    arr_temp = [1]
    
fold_statistics=[]        
# %%
# Creating folds section
for fold in fold_index_set:
    statistics={}
    start_time = time.time()
    print('fold_index =',fold)
    if server == True:
        folder_path = f"""/home/grad/m3yazdan/Github/Yuting_code_modifications/Result/{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers/fold_{fold}"""
        confusion_path = f"""/home/grad/m3yazdan/Github/Yuting_code_modifications/Result/{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers"""
        subfolder_path_individuals = 'teacher'
        subfolder_path_ensemble_learning = f"""/{num_epochs_ensemble_training}ensemble_epochs_lambda_{lambdaa }_learningrate_{learning_rate}/""" 
        logger_path = '/logger.csv'
        process_path = f"""Result/{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers/fold_{fold}"""
    
    else:
        folder_path = f"""C:\\Users\\miyaz\\Documents\\GitHub\\Yuting_code_modifications\\Result\\{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers\\fold_{fold}"""    
        confusion_path =  f"""C:\\Users\\miyaz\\Documents\\GitHub\\Yuting_code_modifications\\Result\\{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers""" 
        subfolder_path_individuals = '\\teacher'
        subfolder_path_ensemble_learning = f"""\\{num_epochs_ensemble_training}ensemble_epochs_lambda_{lambdaa }_learningrate_{learning_rate}""" 
        logger_path = '\\logger.csv'
        process_path = f"""Result\\{dataset}_{architecture}_{architecture_num}_{num_epochs_individual_training}individual_epochs_{num_teachers}teachers\\fold_{fold}"""
    
    if dataset == 'CIFAR':
        dataset_class = CIFAR(fold)
        

    elif dataset == 'MNIST':
        dataset_class = MNIST(fold)
    

    # train ensemble of networks for MNIST with partitioned training data
    print('folder_path',folder_path)
    phase1_start_time = time.time()   
    if phase1_training:

        train_teachers_all(dataset_class,
                       params,
                       arr_temp,
                       folder_path,
                       arrInit=None,
                       num_epochs=num_epochs_individual_training,
                       id_start=0,
                       bool_partition=bool_partition,
                       bool_bagging=bool_bagging,
                       partial_data_usage = partial_data_usage)
        
        plot_training_accuracy_curves(folder_path,num_teachers)
        end_time = time.time()
        b=open(os.path.join(folder_path,'hyper_parametes.txt'),'w')
        b.write("time elapsed in the individual training phase: "+ str(end_time-start_time)+'\n'
                +'server:'+ str(server)+ '\n'
                +'dataset:'+ dataset +'\n'
                +'noisy_logit:'+ str(noisy_logit) +'\n'
                +'partial_data_usage:' + str(partial_data_usage) +'\n'
                +'architecture:' + architecture +'\n'
                +'architecture_num:'+ str(architecture_num)+ '\n'
                +'random_architecture:'+str(random_architecture)+'\n'
                +'num_teachers:' + str(num_teachers) +'\n'
                +'num_epochs_individual_training:'+ str(num_epochs_individual_training) +'\n'
                +'num_epochs_ensemble_training:' + str(num_epochs_ensemble_training) +'\n'
                +'learning_rate:' + str(learning_rate) +'\n'
                +'lambdaa:' + str(lambdaa) +'\n'
                +'temp_limit:' + str(temp_limit) +'\n'
                +'ensemble training batch_size:' + str(ensemble_train_batch_size) +'\n'
                +'ensemble_test_batch_size:'+str(ensemble_test_batch_size)+'\n'
                +'number_of_single_attacked_samples:'+ str(number_of_attacked_samples)+'\n'
                +'number_of_superimposition_attacked_samples: '+ str(number_of_attacked_samples_sup)+'\n'
                +'attack_batch_size:'+str(attack_batch_size)+'\n'
                +'params:' + str(params) +'\n'
                +'temperatures:'+str(arr_temp)+'\n'
                +'nSup: ' + str(nSup) +'\n'
                +'cross_validation: '+ str(cross_validation)+'\n'
                +'fold_index_set:' + str(fold_index_set)+'\n'
                +'fold_index:'+str(fold)+'\n'
                +"bool_partition:" + str(bool_partition)+'\n'
                +"bool_bagging:" +str(bool_bagging)
                )
                
        b.close()
        
        print("Training phase_1 finished _ time elapsed in the individual training phase: ",
              end_time-start_time)
   
    members = load_members(folder_path,
                           subfolder_path_individuals,
                           temp_limit=temp_limit,
                           server=server) 
    #print('summary of the first model after first phase of training:  ', members[0].summary())
        
    n_members = len(members)
    print('number of members:', type(n_members))   
    
    if load_data_model:
        train_data = dataset_class.train_data
        train_labels = dataset_class.train_labels
        test_data = dataset_class.test_data
        test_labels = dataset_class.test_labels
        validation_data = dataset_class.validation_data
        validation_labels = dataset_class.validation_labels  
        
        if partial_data_usage:
            train_data = train_data[:1000]
            train_labels = train_labels[:1000]
            test_data = test_data[:200]
            test_labels = test_labels[:200]
            validation_data = validation_data[:200]
            validation_labels = validation_labels[:200]
            dataset_class.test_data_correct = dataset_class.test_data_correct[:200]
            dataset_class.test_labels_correct = dataset_class.test_labels_correct[:200]
            
        #print('train_data.shape:', train_data.shape)
        #print('train_labels.shape:', train_labels.shape)
        
        # test_data,test_labels = load_batch(data_path+'test_batch.bin')
        # test_labels = test_labels.astype(int)
        # if partial_data_usage:
        #     test_data = test_data[:100]
        #     test_labels = test_labels[:100]
        

        
        # Calculate Rtl and Rll and accuracy and confusion matrix for the ensemble after individual training
        if phase1_training == True:
            # TO Do : Create a function to calculate the statistics
            # for i in range(len(members)):
            #     model = members[i]
            # ens_input = [model.input for model in members]
            tshape = test_data.shape
            tloc = np.zeros(tshape)
            noisy_ens_outputs_test = [model.predict(test_data+np.random.normal(tloc, scale = noise_scale)) for model in members]
            vshape = validation_data.shape
            vloc = np.zeros(vshape)
            noisy_ens_outputs_validation = [model.predict(validation_data+np.random.normal(vloc, scale = noise_scale)) for model in members]
            #Calculate without noisy logits
            ens_outputs_test = [model.predict(test_data) for model in members]
            ens_outputs_validation = [model.predict(validation_data) for model in members]
            
            indvidual_test_acc = []
            individual_validation_acc = []
            # NL means noisy logit
            individual_test_acc_nl = []
            individual_validation_acc_nl =[]
            for nn ,model in enumerate(members):
                predicts_nl_test = np.argmax(noisy_ens_outputs_test[nn], axis=1)
                predicts_nl_validation = np.argmax(noisy_ens_outputs_validation[nn], axis=1)
                predict_test = np.argmax(ens_outputs_test[nn], axis=1)
                predict_validation = np.argmax(ens_outputs_validation[nn], axis=1)
                indv_test_labels = np.argmax(test_labels,axis=1)
                indv_validation_labels = np.argmax(validation_labels,axis=1)
                indvidual_test_acc.append(np.sum(predict_test == indv_test_labels)/len(test_labels))
                individual_validation_acc.append(np.sum(predict_validation == indv_validation_labels)/len(validation_labels))
                individual_test_acc_nl.append(np.sum(predicts_nl_test == indv_test_labels)/len(test_labels))
                individual_validation_acc_nl.append(np.sum(predicts_nl_validation == indv_validation_labels)/len(validation_labels))
           
            h=open(os.path.join(folder_path,'single_statistics.txt'),'w')
            
            h.write('single network test accuracies after individual training:'+ ','.join([str(ind_test_acc) for ind_test_acc in indvidual_test_acc])+'\n'
                    +'single network validation accuracy after individual training::'+ ','.join([str(ind_validation_acc) for ind_validation_acc in individual_validation_acc])+ '\n'
                    +'average of test accuracies:' +str(np.mean(indvidual_test_acc)) +'\n'
                    +'STD of test accuracies:'+str(np.std(indvidual_test_acc)) + '\n'
                    +'average of validation accuracies:' +str(np.mean(individual_validation_acc)) +'\n'
                    +'STD of validation accuracies:'+str(np.std(individual_validation_acc)) + '\n'
                    +'    _______________________________ '+'\n'+'\n'
                    +'Noisy single network test accuracies after individual training:'+ ','.join([str(ind_test_acc_nl) for ind_test_acc_nl in individual_test_acc_nl])+'\n'
                    +'Noisy single network validation accuracy after individual training::'+ ','.join([str(ind_validation_acc_nl) for ind_validation_acc_nl in individual_validation_acc_nl])+ '\n'
                    +'Noisy average of test accuracies:' +str(np.mean(individual_test_acc_nl)) +'\n'
                    +'Noisy STD of test accuracies:'+str(np.std(individual_test_acc_nl)) + '\n'
                    +'Noisy average of validation accuracies:' +str(np.mean(individual_validation_acc_nl)) +'\n'
                    +'Noisy STD of validation accuracies:'+str(np.std(individual_validation_acc_nl)) + '\n'
                    +'    _______________________________ '+'\n'+'\n')   
            
            h.close()
            statistics['single models validation accuracies'] = individual_validation_acc  
            statistics['single models test accuracies'] = indvidual_test_acc
            statistics['Average of single models validation accuracies'] = np.mean(individual_validation_acc)
            statistics['Average of single models test accuracies'] = np.mean(indvidual_test_acc)
            
            # x_list_input = [test_data for i in range(n_members)]
            y_list_output_test = [test_labels for i in range(n_members)]
            y_list_output_validation = [validation_labels for i in range(n_members)]
            
            noisy_main_accuracy_test = main_acc(noisy_ens_outputs_test,test_labels)
            noisy_main_accuracy_validation = main_acc(noisy_ens_outputs_validation,validation_labels)
            noisy_confusion_matrix_validation = ensemble_confusion_matrix(noisy_ens_outputs_validation , validation_labels)
            noisy_acc_confusion = np.trace(noisy_confusion_matrix_validation)/np.sum(noisy_confusion_matrix_validation)
            noisy_RTL=rtl_symmetric(noisy_ens_outputs_test,y_list_output_test)
            RTL=rtl_symmetric(ens_outputs_test,y_list_output_test)
            
            if architecture != 'single':
                noisy_RLL=rll(noisy_ens_outputs_test)
                RLL=rll(ens_outputs_test)
            else:
                RLL = None
                noisy_RLL = None
            if noisy_logit:
                main_accuracy_test, correct_indices_test = main_acc(noisy_ens_outputs_test,
                                                                    test_labels,
                                                                    return_correct_indices=True)
            else:    
                main_accuracy_test, correct_indices_test = main_acc(ens_outputs_test,
                                          test_labels,
                                          return_correct_indices=True)
            
            #print('shape of correct indices:', correct_indices_test.shape)
            #print('number of correct indices:', np.sum(correct_indices_test))
            #print('first correct indices:', correct_indices_test[:10])
            
            main_accuracy_validation = main_acc(ens_outputs_validation,validation_labels)
            confusion_matrix_validation = ensemble_confusion_matrix(ens_outputs_validation , validation_labels)
            # To Do: confusion_matrix_test = ensemble_confusion_matrix(ens_outputs_test , test_labels)
            
            acc_confusion = np.trace(confusion_matrix_validation)/np.sum(confusion_matrix_validation)
            print('Rtl is:',RTL,'Rll is:',RLL, 'main accuracy of test data is:', main_accuracy_test ,
                  'main accuracy of validation data is:', main_accuracy_validation,
                  '\n', 'validation confusion matrix is', confusion_matrix_validation,
                  '\n', 'accuracy according to confusion matrix is', acc_confusion)
            
            confusion_acc_list.append(acc_confusion)
            noisy_confusion_acc_list.append(noisy_acc_confusion)
            b=open(os.path.join(folder_path,'ensemble_statistics.txt'),'w')
            b.write('Rtl after individual training:'+ str(RTL) +'\n'
                    +'Rll after individual training:' + str(RLL) +'\n'
                    +'ensemble test accuracy after individual training:'+ str(main_accuracy_test) +'\n'
                    +'ensemble validation accuracy after individual training:'+ str(main_accuracy_validation)+ '\n'
                    +'accuracy according to confusion matrix:' +str(acc_confusion)
                    +'    _______________________________ '+'\n'+'\n'
                    +'Rtl after individual training for noisy data:'+ str(noisy_RTL) +'\n'
                    +'Rll after individual training for noisy data:' + str(noisy_RLL) +'\n'
                    +'ensemble test accuracy after individual training for noisy data:'+ str(noisy_main_accuracy_test) +'\n'
                    +'ensemble validation accuracy after individual training for noisy data:'+ str(noisy_main_accuracy_validation)+ '\n'
                    +'accuracy according to confusion matrix for noisy data:' +str(noisy_acc_confusion)
                    )   
            
            b.close()
            np.savetxt(os.path.join(folder_path,"confusion_matrix.csv"),np.around(confusion_matrix_validation,2),delimiter=',')
            np.savetxt(os.path.join(folder_path,"noisy_confusion_matrix.csv"),np.around(noisy_confusion_matrix_validation,2),delimiter=',')
            statistics['Validation ensemble clean accuracy phase 1'] = main_accuracy_validation 
            statistics['Test ensemble clean accuracy phase 1'] = main_accuracy_test
            statistics['Rtl after individual training'] = tf.keras.backend.get_value(RTL)
            statistics['Rll after individual training'] = tf.keras.backend.get_value(RLL)
    phase1_end_time = time.time()
    phase1_time = phase1_end_time - phase1_start_time    
    statistics['phase1 time'] = phase1_time 
    print('phase1 time', phase1_time)
    phase2_start_time = time.time()
    if phase2_training:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,                
                                            decay=1e-6, momentum=0.9, nesterov=True)
        
        model = define_stacked_model_simple(members)
         
        ensemble_training(train_data, 
                          train_labels,
                          ensemble_train_batch_size,
                          num_epochs_ensemble_training,
                          lambdaa,
                          optimizer,
                          model,
                          folder_path,
                          subfolder_path_ensemble_learning,
                          logger_path,
                          n_members=n_members,
                          test_data=validation_data,
                          test_labels=validation_labels,
                          ensemble_test_batch_size=ensemble_test_batch_size,
                          server=server,
                          pang=pang
                          )
    
            
    # plotting(folder_path,subfolder_path_ensemble_learning,logger_path)
    
    last_epoch_num = str((num_epochs_ensemble_training)-1)
    phase2_end_time = time.time()
    phase2_time = phase2_end_time - phase2_start_time    
    statistics['phase2 time'] = phase2_time 
    print('phase2 time', phase2_time)
    # Breaking the stacked model
    phase3_start_time = time.time()
    if phase3_breaking:
        
        model_path = os.path.join(folder_path+subfolder_path_ensemble_learning,'epoch_'+last_epoch_num)
        
        print('stacked_model_path', model_path)
        model = load_model(model_path)
        
        broken_members = break_stacked_model(stacked_model=model,
                            n_members=n_members,
                            train_temp=arr_temp,
                            input_shape=break_input_shape
                            )
        
        break_path = os.path.join(folder_path+subfolder_path_ensemble_learning,
                                  'epoch_'+last_epoch_num+'_broken_models')   
        for i,member in enumerate(broken_members):
            broken_model_name = f'teacher_'+str(i)
            member.save(os.path.join(break_path,broken_model_name))
            description_path = os.path.join(folder_path,'teacher_'+str(i))
            #print('sourcepath: ',os.path.join(description_path,'model_description.txt'))
            shutil.copy(os.path.join(description_path,'model_description.txt'),
                        os.path.join(break_path,broken_model_name))
            
    
        #broken_models= load_members(break_path, subfolder_path_individuals, temp_limit, server=server)
        #print('summary of the first models after breaking the models: ', broken_models[1].summary())
        #To DO: Record the statistics of each broken model and also the ensemble of them
    
    last_epoch_num = str((num_epochs_ensemble_training)-1)        
    break_path = os.path.join(folder_path+subfolder_path_ensemble_learning,
                                      'epoch_'+last_epoch_num+'_broken_models') 
    break_process_path = os.path.join(process_path+subfolder_path_ensemble_learning,
                                      'epoch_'+last_epoch_num+'_broken_models') 
    phase3_end_time = time.time()
    phase3_time = phase3_end_time - phase3_start_time    
    statistics['phase3 time'] = phase3_time 
    print('phase3 time', phase3_time)
# %%  Attacking the models 
    
    if Attack_phase1_models:
        members = load_members(folder_path, subfolder_path_individuals, 
                               temp_limit, server=server)
    else:
        members = load_members(break_path , subfolder_path_individuals, 
                               temp_limit, server=server)
    if noisy_logit:
        tloc = np.zeros(test_data.shape)
        noisy_ens_outputs_test = [model.predict(test_data+np.random.normal(tloc,
                                                                            scale = noise_scale))
                                  for model in members]
        main_accuracy_test, correct_indices_test = main_acc(noisy_ens_outputs_test,
                                                            test_labels,
                                                            return_correct_indices=True)
    else:
        ens_outputs_test = [model.predict(test_data) for model in members]
        main_accuracy_test, correct_indices_test = main_acc(ens_outputs_test,
                                                            test_labels,
                                                            return_correct_indices=True)
    
    arrfilenames = []
    dataset_class.test_data_correct = dataset_class.test_data_correct[correct_indices_test]
    dataset_class.test_labels_correct = dataset_class.test_labels_correct[correct_indices_test]
    # print('size of dataset_class.test_data_correct: ',
    #       dataset_class.test_data_correct.shape)
    # print('size of dataset_class.test_labels_correct: ',
    #       dataset_class.test_labels_correct.shape)
    
    for i in range(num_teachers):
        if Attack_phase1_models == True:
            arrfilenames.append(os.path.join(process_path,f'teacher_'+str(i)))
        else:
            arrfilenames.append(os.path.join(break_process_path,f'teacher_'+str(i)))
                                       
    print('arr file names:',arrfilenames)
        
    if Attack_phase1_models == True:
        if noisy_logit == True:
            strTest = os.path.join(folder_path,'single_attack_noisy')
        else:
            strTest = os.path.join(folder_path,'single_attack_noiseless')
    elif Attack_phase1_models == False:
        if noisy_logit == True:
            strTest = os.path.join(break_path,'single_attack_noisy')
        else:
            strTest = os.path.join(break_path,'single_attack_noiseless')
    
    #test transferability    
    phase4_single_start_time = time.time()
    if phase4_single_attack:
        makeDir(strTest)
        c=open(os.path.join(strTest,'attack_parameters.txt'),'w')
        c.write('attack_max_iterations: '+ str(attack_max_iterations)+'\n'
                +'attack_learning_rate: '+str(attack_learning_rate)+'\n'
                +'Dataset model: '+str(dataset_model)+'\n'
                +'number_of_single_attacked_samples:'+ str(number_of_attacked_samples)+'\n'
                +'attack_batch_size:'+str(attack_batch_size)+'\n'
                )                
        c.close()
        
        for iStart in range(nstart_attack,
                            nstart_attack+number_of_attacked_samples,
                            attack_batch_size):
            if Attack_phase1_models == True:
                    folder_image_save = folder_path
            else:
                    folder_image_save = break_path
    

            test_ensemble_noisylogit_superimposed_all_byinput(dataset_class,                                                     
                                                              dataset_model,
                                                              arrfilenames,
                                                              folder_image_save,
                                                              params,
                                                              temps=arr_temp,
                                                              strTest=strTest,
                                                              samples=attack_batch_size,
                                                              start=iStart,
                                                              plotIMG=plot_img,
                                                              strImgFolder=folder_image_save,
                                                              noisy_logit = noisy_logit,     
                                                              attack_max_iterations=attack_max_iterations,
                                                              attack_learning_rate=attack_learning_rate)
        
# %%  Process results for transferability test
    if phase5_process_single_attack:

        #print('noisy_logit:', noisy_logit)
        if Attack_phase1_models == True:
            if noisy_logit == True:
                strTest = os.path.join(folder_path,'single_attack_noisy')
            else:
                strTest = os.path.join(folder_path,'single_attack_noiseless')
        elif Attack_phase1_models == False:
            if noisy_logit == True:
                strTest = os.path.join(break_path,'single_attack_noisy')
            else:
                strTest = os.path.join(break_path,'single_attack_noiseless')
        makeDir(strTest)
        strResult = os.path.join(strTest,'final_results.txt')
        #print('strResult:', strResult ,'strTest:' ,strTest  )
        stradvvote='adv_vote'
        strorigvote='orig_vote'
        str2target='correct_to_target'
        str2other='correct_to_other'
        strdistort='Distortion'
        strnorm='Image_norm'
        strinput='Input'
        strtarget='Target'
        if Attack_phase1_models == True:           
            strfolder1 = process_path
        else:
            strfolder1 = break_process_path 

        for i in range(nstart_attack+1 ,nstart_attack+number_of_attacked_samples+1):
            tmp = str(i)
            if noisy_logit == True:
                if Attack_phase1_models == True:
                    strfileprefix1 = os.path.join(process_path,'test_img_allmodel_noisy_logits_'+tmp)
                else:
                    strfileprefix1 = os.path.join(break_process_path,'test_img_allmodel_noisy_logits_'+tmp)

            else:
                if Attack_phase1_models == True:
                    strfileprefix1 = os.path.join(process_path,'test_img_allmodel_noiseless_'+tmp)                
                else:
                    strfileprefix1 = os.path.join(break_process_path,'test_img_allmodel_noiseless_'+tmp)                

                    
            print('break_process_path:',break_process_path)
            print('strfileprefix1:',strfileprefix1)
            
            strfilename = os.path.join(strfileprefix1,'testoutput_'+tmp+'_summary.txt')
            #print('strfilename:',strfilename)
            if noisy_logit == True:
                stats_add = os.path.join(strfolder1 ,'single_attack_noisy')
            if noisy_logit == False:
                stats_add = os.path.join(strfolder1 ,'single_attack_noiseless')
            makeDir(stats_add)
            strFileout =os.path.join(stats_add,'summary_stats'+str(i)+'.txt')
            #print('strFileout:',strFileout)
            #this creates a summary file containing statistics of the raw output file generated from test_ensemble_noisylogit_superimposed_all_byinput
            get_transfer_stats(strfilename,strFileout=strFileout)
        
        #treshhold_list = [1,0.75,0.5,0.25,0.10]
        treshhold_list = [0.9,0.75,0.5,0.25]
        for treshhold in treshhold_list:
            out1,out2,out3,out4,out5,out6,out7 = single_process(strfolder1,strResult,treshhold,num_single_adv_examples,noisy_logit)
            if treshhold == 0.5:
                statistics['Accuracy on adversarial examples'] = out1
                statistics['Correct_to_target percentage'] = out2
                statistics['Correct_to_others percentage'] = out3
                statistics['average perturbation for unchanged'] = out4
                statistics['average perturbation for correct to target'] = out5
                statistics['average perturbation for correct to other'] = out6
                statistics['accuracy on original of adversarial examples'] = out7
                               
        #I added this but I am not sure if it works: process_all_results(strTest,bSupImp=False)
        #this groups the stats from the above summary files by different distortion bin sizes
        #get_stats_by_bucket_arrfiles(arrfilenames1, 20)
        overal_add = os.path.join(stats_add,'overal.txt')
        get_min_max_details(overal_add,stats_add)
        
        if attacked_single_statistics:
            individual_statistics(strTest, num_teachers,num_single_adv_examples,noisy_logit)
        
    #test superimposition of 2 or 3
    #Moved to top
    #nSup = 2 # 3
    phase4_single_end_time = time.time()
    phase4_single_time = phase4_single_end_time - phase4_single_start_time    
    statistics['phase4 single attack time'] = phase4_single_time 
    print('phase4 signle attack time', phase4_single_time)
    phase4_sup_start_time = time.time()
    if phase4_sup_attack:
        if Attack_phase1_models == True:           
            strfolder = folder_path
            strfolder1 = process_path
        else:
            strfolder = break_path
            strfolder1 = break_process_path 

        for j in range(nstart_attack,nstart_attack+number_of_attacked_samples_sup,attack_batch_size):
            if noisy_logit == False:
                
                folder_image_save = os.path.join(strfolder,'sup'+str(nSup)+'_test_'+str(j+1)+'-'+str(j+1)+'_noiseless/')
                folder_image_save = makeDir(folder_image_save)
                file_output = os.path.join( folder_image_save,'testoutput_'+str(j+1)+'-'+str(j+1)+'.txt')
                summary_path = os.path.join(strfolder,'sup_'+str(nSup)+'attack_noiseless')
                file_output_summary = os.path.join(summary_path,'testoutput_'+str(j+1)+'-'+str(j+1)+'_summary.txt')
                
            
            if noisy_logit == True:
                
                folder_image_save = os.path.join(strfolder,'sup'+str(nSup)+'_test_'+str(j+1)+'-'+str(j+1)+'_noisy/')
                folder_image_save = makeDir(folder_image_save)
                file_output = os.path.join( folder_image_save,'testoutput_'+str(j+1)+'-'+str(j+1)+'.txt')
                summary_path = os.path.join(strfolder,'sup_'+str(nSup)+'attack_noisy')
                file_output_summary = os.path.join(summary_path,'testoutput_'+str(j+1)+'-'+str(j+1)+'_summary.txt')
            makeDir(summary_path)
            d=open(os.path.join(summary_path,'attack_parameters.txt'),'w')
            d.write('attack_max_iterations: '+ str(attack_max_iterations)+'\n'
                    +'attack_learning_rate: '+str(attack_learning_rate)+'\n'
                    +'Dataset model: '+str(dataset_model)+'\n'
                    +'number_of_superimposition_attacked_samples:'+ str(number_of_attacked_samples_sup)+'\n'
                    +'attack_batch_size:'+str(attack_batch_size)+'\n'
                    )                
            d.close()  
            
            
            test_ensemble_noisylogit_superimposed_opt(data=dataset_class,
                                                      datamodel=dataset_model,
                                                      arr_file_name=arrfilenames,
                                                      params=params,
                                                      temps=arr_temp,
                                                      file_name_out=file_output,
                                                      file_name_summary=file_output_summary,
                                                      summary_path=summary_path,
                                                      nAttacked=nSup,
                                                      samples=attack_batch_size,
                                                      start=j,
                                                      plotIMG=plot_img,
                                                      strImgFolder=folder_image_save,
                                                      noisy_logit=noisy_logit,    
                                                      attack_max_iterations=attack_max_iterations,
                                                      attack_learning_rate=attack_learning_rate)

      
    # Process the result
    #process results for superimposition attacks
    if phase5_process_sup_attack:
        #if the tests were run in small batches, the output summary txt files should be put a single folder
        if Attack_phase1_models == True:           
            strfolder = folder_path
            strfolder1 = process_path
        else:
            strfolder = break_path
            strfolder1 = break_process_path 
        if noisy_logit == True:
            if Attack_phase1_models == True:
                strTest = os.path.join( strfolder,'sup_'+str(nSup)+'attack_noisy/')
            else:
                strTest = os.path.join( strfolder,'sup_'+str(nSup)+'attack_noisy/')
        if noisy_logit == False :   
            if Attack_phase1_models == True:
                strTest = os.path.join( strfolder,'sup_'+str(nSup)+'attack_noiseless/')
            else:
                strTest = os.path.join( strfolder,'sup_'+str(nSup)+'attack_noiseless/')
        print('break_process_path:',break_process_path)        
        makeDir(strTest)
        #process the files (generated from test_ensemble_noisylogit_superimposed_opt) in the above folder into a table, 
        #will generate a master.txt file
        process_all_results(strTest, bSupImp=True)        
        #extract stats for superimposition attacks
        strMaster = os.path.join(strTest,'master.txt')
        strResult = os.path.join(strTest,'final_results.txt')
        # print("Str master: ", strMaster)
        num_sup_attack = number_of_attacked_samples_sup*9
        
        strTeacherPrefix = process_path.lower()
        # arrTeachers = [strTeacherPrefix+'teacher_'+str(i) for i in range(num_teachers)]           
        arrTeachers = []
        for i in range(num_teachers):
            strTeacherPrefix = os.path.join(strfolder1,'teacher_'+str(i))
            strTeacherPrefix = strTeacherPrefix.lower()
            strTeacherPrefix = strTeacherPrefix.replace(os.path.sep,'')
            strTeacherPrefix = strTeacherPrefix.replace('.','')
            
            arrTeachers.append(strTeacherPrefix)
        # print('strMaster:',strMaster)
        print('arrTeachers:',arrTeachers)
        # print('strResult:',strResult)
        #extract statistics from the master.txt file
        unchanged_acc , target_acc , other_acc , distort_input , distort_target , distort_other = extract_stats(strMaster, arrTeachers,strResult,num_sup_attack)
        statistics['accuracy of the ensemble after sup attack'] = unchanged_acc
        statistics['corret to target percentage after sup attack'] = target_acc
        statistics['corret to other percentage after sup attack'] =  other_acc
        statistics['average perturbation for unchanged sup attack'] = distort_input
        statistics['average perturbation for correct to target sup attack'] = distort_target
        statistics['average perturbation for correct to other sup attack'] = distort_other 
    fold_statistics.append(statistics) 
    phase4_sup_end_time = time.time()
    phase4_sup_time = phase4_sup_end_time - phase4_sup_start_time    
    statistics['phase4 sup attack time'] = phase4_sup_time 
    print('phase4 sup attack time', phase4_sup_time)
    df_final_statistics = pd.DataFrame(fold_statistics)
    file_name = 'final_statistics_Noise_'+str(noisy_logit)+'_phase1_'+str(phase1_training)+'_phase2_'+str(phase2_training)+'_phase3_'+str(phase3_breaking)+'_phase4_single_'+str(phase4_single_attack)+'_phase4_sup_'+str(phase4_sup_attack)+'_attack_phase1_'+str(Attack_phase1_models)+'.xlsx'
    path = os.path.join(confusion_path,file_name)
    df_final_statistics.to_excel(path)
# %%
# Printing fold statistics 
 
if phase1_training== True:
    c=open(os.path.join(confusion_path,'confusion.txt'),'w')
    c.write('list of accuracies according to confusion matrix:'+str(confusion_acc_list)+'\n'
            +'average of accuracies:'+str(np.mean(confusion_acc_list))+'\n'
            +'standard deviation of accuracies:'+ str(np.std(confusion_acc_list)))
    c.close()
    d=open(os.path.join(confusion_path,'noisy_confusion.txt'),'w')
    d.write('list of accuracies according to confusion matrix for noisy data:'+str(noisy_confusion_acc_list)+'\n'
            +'average of accuracies for noisy data:'+str(np.mean(noisy_confusion_acc_list))+'\n'
            +'standard deviation of accuracies for noisy data:'+ str(np.std(noisy_confusion_acc_list)))
    d.close()   
    
# df_final_statistics = pd.DataFrame(fold_statistics, index=['fold 5','fold 4','fold 3','fold 2','fold 1'])
# file_name = 'final_statistics_phase1_'+str(phase1_training)+'_phase2_'+str(phase2_training)+'_phase3_'+str(phase3_breaking)+'_phase4_single_'+str(phase4_single_attack)+'_phase4_sup_'+str(phase4_sup_attack)+'_attack_phase1_'+str(Attack_phase1_models)+'.xlsx'
# path = os.path.join(confusion_path,file_name)
# df_final_statistics.to_excel(path)

        

