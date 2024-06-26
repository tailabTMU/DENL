import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from setup_cifar import CIFAR, CIFARModel
import multiprocessing as mp
from l2_attack import CarliniL2
from random import sample
import pandas as pd

def makeDir(strPath):
    if not os.path.exists(strPath):
        try:
            os.makedirs(strPath)
        except:
            print('skip makedir')
    return strPath

## original code by Nicholas Carlini
def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def plotCIFAR(img, strImageFolder='', strImageName=''):
    timg = (img+0.5)*255
    timg = np.array(timg, dtype='uint8')
    timg = timg.reshape(32,32,3)
    plt.imshow(timg)
    if strImageFolder != '':
        plt.savefig(strImageFolder+strImageName)
    else:
        plt.show()
    plt.close()

def plotMNIST(img, strImageFolder='', strImageName=''):
    #turn the attacked image from Carlini to a standard format image
    timg = (img+0.5)*255
    #change the pixel values from floating point to integer
    timg = np.array(timg, dtype='uint8')
    #change the vector form to an 2D image
    timg = timg.reshape(28,28)
    plt.imshow(timg, cmap='gray')
    if strImageFolder != '':
        plt.savefig(strImageFolder+strImageName)
    else:
        plt.show()
    plt.close()


def showCifar(img):
    timg = (img+0.5)*255
    plt.imshow(timg)   
    plt.show()
    plt.close()

## original code by Nicholas Carlini
def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    input_labels = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels_correct.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels_correct[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data_correct[start+i])
                targets.append(np.eye(data.test_labels_correct.shape[1])[j])
                input_labels.append(np.argmax(data.test_labels_correct[start+i]))
        else:
            inputs.append(data.test_data_correct[start+i])
            targets.append(data.test_labels_correct[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    if targeted:
        input_labels = np.array(input_labels)
        return inputs , targets , input_labels
    else:
        return inputs, targets

#%% superimposition attack
def test_ensemble_noisylogit_superimposed_opt(data,
                                              datamodel,
                                              arr_file_name,
                                              params,
                                              temps,
                                              file_name_out,
                                              file_name_summary,
                                              summary_path,
                                              nAttacked,
                                              samples=1,
                                              start=0,
                                              plotIMG=plotCIFAR,
                                              strImgFolder='',
                                              noisy_logit=True,
                                              attack_max_iterations=1000,
                                              attack_learning_rate=0.01):
    arr_models = []
    arr_attack = []
    arr_adv = []
    f_summary = open(file_name_summary, 'w')
    f_out = open(file_name_out, 'w')
    
    #generate samples and targets
    # TO DO : add input_labels to output arguments of the function
    inputs, targets, input_labels= generate_data(data,
                                    samples=samples,
                                    targeted=True,
                                    start=start,
                                    inception=False)
    
    print("shape of inputs: ", inputs.shape)
    print("shape of targets: ", targets.shape)
    
    timestart = time.time()
    f_summary.write('Begin Timestamp: '+str(timestart)+'\n')
    f_summary.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
    f_out.write('Begin Timestamp: '+str(timestart)+'\n')
    f_out.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
    with tf.compat.v1.Session() as sess:
        #restore ensemble networks from files
        for ii , file_model in enumerate(arr_file_name):
            temp = temps[ii]
            params_ii = params[ii%len(params)]
            #instanciating the model 
            model = datamodel(params=params_ii,
                              temp=temp,
                              restore=file_model,
                              session=sess)
            #make an array of the ensemble models
            arr_models.append(model)
            start_time = time.time()
        #generate attacks for each network in the ensemble for each sample-target pair
        for i in range(len(arr_models)):
            attack = CarliniL2(sess,
                               arr_models[i],
                               batch_size=9*samples,
                               max_iterations=attack_max_iterations,
                               binary_search_steps=5,
                               learning_rate=attack_learning_rate,
                               initial_const=0.01,
                               confidence=0)
            arr_attack.append(attack)
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            arr_adv.append(adv)
            end_time = time.time()
            print('time:'+ str(end_time-start_time))
        tmp_Imposed_all = []
        for j in range(len(inputs)):
            start_time2 = time.time()
            print('***** J:',j)
            
            arr_distortion = [np.sum((adv[j:j+1]-inputs[j:j+1])**2)**.5 for adv in arr_adv]
            #f_summary.write('all distortions: '+';'.join([str(dist) for dist in arr_distortion])+'\n')
            #get the attacks with the smallest distortiohs
            arr_indices = np.array(arr_distortion).argsort()[0:nAttacked]
            strAttInd = '-'.join([str(ind) for ind in arr_indices])
            arrAttackedImage = [arr_adv[ind][j:j+1] for ind in arr_indices]
            #get superimposition of the chosen attacks
            tmpImposed = get_superimposed(inputs[j:j+1], arrAttackedImage)
            # if tmpImposed.ndim > 4:
            #     tmp_Imposed_all.append(np.squeeze(tmpImposed))
            # else:
            tmp_Imposed_all.append(tmpImposed)
            end_time2 = time.time()
            print('time 2:'+ str(end_time2-start_time2))
            #strImageInput = 'a'+strAttInd+'_input'+str(j)
            #print('input image: +\n')
            #save the sample image (without distortion)
            # plotIMG(inputs[j], strImgFolder, strImageInput+'.png')
            # show(inputs[j])
            #f_summary.write('Target: '+str(np.argmax(targets[j]))+'\n')
            
            #for a in range(nAttacked):
                #strImageAdv = strImageInput+'_adv'+str(arr_indices[a])
                #print('adv image'+str(arr_indices[a])+': \n')
                #save each attacked image in the superimposition
                # plotIMG(arrAttackedImage[a], strImgFolder, strImageAdv+'.png')
                # show(arrAttackedImage[a])
                #adv_distortion = np.sum((arrAttackedImage[a]-inputs[j:j+1])**2)**.5
                #f_summary.write(strImageAdv+' perturbation: '+str(adv_distortion)+'\n')

            #print('adv image superimposed: +\n')
            #strImageSup = strImageInput+'_advsup'
            #save the superimposed image
            # plotIMG(tmpImposed, strImgFolder,strImageSup+'.png')
            # show(tmpImposed)

            
   
        arr_class_orig_byind = []  
        arr_class_adv_byind = []
        arr_classification = [[]for k in range(len(arr_file_name))] 
        for k in range(len(arr_file_name)):
            x = tf.compat.v1.placeholder(tf.float32, (None,
                                                      arr_models[k].image_size,
                                                      arr_models[k].image_size,
                                                      arr_models[k].num_channels))
            # model_original = arr_models[k].predict(x)
            
            original_predictions = get_prediction_array(sess,
                                                        arr_models[k],
                                                        x,
                                                        list(inputs))
            #print('original predictions:',original_predictions)
            #for each network, classify all original inputs
            arr_class_orig_byind.append(original_predictions)
            #for each network, classify all attacked images targeted at a specific network
            f = get_prediction_array(sess,
                                    arr_models[k],
                                    x,
                                    tmp_Imposed_all)
            arr_class_adv_byind.append(f)
            #classify the original input
            #class_original = sess.run(model_original, {x: inputs[j:j+1]})
            #classify the superimposed image
            #class_adv = sess.run(model_original, {x: tmpImposed})
            print("Prediction array:", f)

               
        for j in range(len(inputs)):
            arr_distortion = [np.sum((adv[j:j+1]-inputs[j:j+1])**2)**.5 for adv in arr_adv]
            arr_indices = np.array(arr_distortion).argsort()[0:nAttacked]
            arrAttackedImage = [arr_adv[ind][j:j+1] for ind in arr_indices]
            f_summary.write('all distortions: '+';'.join([str(dist) for dist in arr_distortion])+'\n')
            f_summary.write('Target: '+str(np.argmax(targets[j]))+'\n')
            strImageInput = 'a'+strAttInd+'_input'+str(j) 
            for a in range(nAttacked):
                strImageAdv = strImageInput+'_adv'+str(arr_indices[a])
                #print('adv image'+str(arr_indices[a])+': \n')
                #save each attacked image in the superimposition
                # plotIMG(arrAttackedImage[a], strImgFolder, strImageAdv+'.png')
                # show(arrAttackedImage[a])
                adv_distortion = np.sum((arrAttackedImage[a]-inputs[j:j+1])**2)**.5
                f_summary.write(strImageAdv+' perturbation: '+str(adv_distortion)+'\n')
            strImageSup = strImageInput+'_advsup'
            #save the superimposed image
            #plotIMG(tmpImposed, strImgFolder,strImageSup+'.png')
            #show(tmpImposed)
            for k in range(len(arr_file_name)):
                f_out.write(arr_file_name[k]+" Classification: "+str(arr_class_adv_byind[k][j])+'\n')
                f_summary.write(arr_file_name[k].lower()+" original Classification: "+
                                str(arr_class_orig_byind[k][j])+'\n')
                f_summary.write(arr_file_name[k].lower()+" adv Classification: "+
                                str(arr_class_adv_byind[k][j])+'\n')
                arr_classification[k].append(arr_class_adv_byind[k][j])
                
            total_distortion = np.sum((tmp_Imposed_all[j]-inputs[j:j+1])**2)**.5
            img_norm =  np.sum((inputs[j:j+1])**2)**.5
            #write the final vote
            all_model_class = [arr_classification[k][j] for k in range(len(arr_file_name))]
            f_summary.write('Final vote: '+str(np.bincount(all_model_class).argmax())+'\n')
            print("Total distortion: ", total_distortion)
            print('Image norm: ', img_norm)
            f_out.write("Total distortion: "+str(total_distortion)+'\n')
            f_out.write('Image norm: '+str(img_norm)+'\n')
            f_summary.write("Total distortion: "+str(total_distortion)+'\n')
            f_summary.write('Image norm: '+str(img_norm)+'\n')

    timeend=time.time()
    f_summary.write('End Timestamp: '+str(timeend)+'\n')
    f_out.write('End Timestamp: '+str(timeend)+'\n')
    f_out.close()
    f_summary.close()  
    end_time2 = time.time()
    print('time for final fors:', end_time-timestart)
    print("total time:",end_time2-timestart)
    attack_time = os.path.join(strImgFolder,'attack_time.txt')
    s_summary = open(attack_time, 'a')
    s_summary.write('attack time:'+ str(end_time2-timestart)+'\n') 
    s_summary.close()
    

#%% check transferability
def test_ensemble_noisylogit_superimposed_all_byinput(data,
                                                      datamodel,
                                                      arr_file_name,
                                                      folder_image_save,
                                                      params,
                                                      temps,
                                                      strTest,
                                                      samples=1,
                                                      start=0,
                                                      plotIMG=plotCIFAR,
                                                      strImgFolder='',
                                                      noisy_logit= True,
                                                      attack_max_iterations=1000,
                                                      attack_learning_rate=0.01):
    arr_attack = []
    arr_adv = []
    arr_models = []  
    inputs, targets, input_labels  = generate_data(data, 
                                    samples=samples,
                                    targeted=True,
                                    start=start,
                                    inception=False)
    
    
    print("shape of inputs: ", inputs.shape)
    print("label of inputs: ", input_labels)   
    print("shape of targets: ", targets.shape)
    print("shape of input_labels: ", input_labels.shape)
    timestart = time.time()

    dict_class_orig = [[] for ind in range(len(arr_file_name))]
    dict_class_adv = [[] for ind in range(len(arr_file_name))]
    with tf.compat.v1.Session() as sess:
        #restore ensemble networks from files
        for ii , file_model in enumerate(arr_file_name):
            temp = temps[ii]
            print('file model:',file_model)
            params_ii = params[ii%len(params)]
            print('params_ii:',params_ii)
            #instanciating the model 
            model = datamodel(params=params_ii,
                              temp=temp,
                              restore=file_model,
                              session=sess)
            #make an array of the ensemble models
            arr_models.append(model)
            
        #print('number of models:',len(arr_models))
        #generate attacks for each network in the ensemble for each sample-target pair
        start_time = time.time()
        for i in range(len(arr_models)):
            attack = CarliniL2(sess,
                               arr_models[i],
                               batch_size=9*samples,
                               max_iterations=attack_max_iterations,
                               binary_search_steps=5,
                               learning_rate=attack_learning_rate,
                               initial_const=0.01,
                               confidence=0)
            arr_attack.append(attack)
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            #print('shape of adv:', adv.shape)
            timeend = time.time()
            arr_adv.append(adv)
        end_time = time.time()
        print('time:'+ str(end_time-start_time))
        print('attacks done.')
        # get_all_cpu_mem_info()
        #pool = mp.Pool(mp.cpu_count())
        # Mina: commenting below lines to not plot the images
        arr_target = [str(np.argmax(targets[j])) for j in range(len(inputs))]
        arr_input = [str(input_labels[j]) for j in range(len(inputs))]
        arr_img_input = ['input'+arr_input[j]+'_target'+arr_target[j] for j in range(len(inputs))]
        #save each input image
        #strImgFolder1 = os.path.join(strImgFolder,'image_')
        #imgs_input_results = [pool.apply(plotIMG, args=(inputs[j],strImgFolder1,arr_img_input[j]+'.png')) for j in range(len(inputs))]
        # for j in range(len(inputs)):
        #     plotIMG(inputs[j],strImgFolder1,arr_img_input[j]+'.png')
        arr_feeds = [inputs[j:j+1] for j in range(len(inputs))]
        print('starting quries')
        # get_all_cpu_mem_info()
        start_time=time.time()
      
        for ind in range(len(arr_models)):
            arrAttackedImage = [arr_adv[ind][j:j+1] for j in range(len(inputs))]
            x = tf.compat.v1.placeholder(tf.float32,(None,
                                                     arr_models[ind].image_size,
                                                     arr_models[ind].image_size,
                                                     arr_models[ind].num_channels))
            #save each attacked image
            # Mina: commenting below lines to not plot the images
            #arr_img_adv = [arr_img_input[j]+'_adv'+str(ind) for j in range(len(inputs))]
            #imgs_adv_results = [pool.apply(plotIMG, args=(arrAttackedImage[j],strImgFolder1,arr_img_adv[j]+'.png')) for j in range(len(inputs))]
            # for j in range(len(inputs)):
            #     plotIMG(arrAttackedImage[j],strImgFolder1,arr_img_adv[j]+'.png')

            arr_class_adv_byind = []
            arr_class_orig_byind = []
            for k in range(len(arr_models)):
                original_predictions = get_prediction_array(sess,
                                                            arr_models[k],
                                                            x,
                                                            list(inputs))
                #print('original predictions:',original_predictions)
                #for each network, classify all original inputs
                arr_class_orig_byind.append(original_predictions)
                #for each network, classify all attacked images targeted at a specific network
                arr_class_adv_byind.append(get_prediction_array(sess,
                                                                arr_models[k],
                                                                x,
                                                                list(arr_adv[ind])))
            dict_class_orig[ind] = arr_class_orig_byind
            dict_class_adv[ind] = arr_class_adv_byind
            # get_all_cpu_mem_info()
    end_time =  time.time()
    print('time of two for loops', end_time-start_time)
    print('starting write output')
    # get_all_cpu_mem_info()
    columns=['input_label','Target']
    columns+=['original_classification_'+str(i) for i in range(len(arr_models))]
    columns+=['adv_classification_'+str(i) for i in range(len(arr_models))]
    df = pd.DataFrame(columns= columns)
    norm=[]
    distort=[]
    input_image_counter = 0
    counter = 0
    start_time = time.time()
    if noisy_logit is True:
        noise_status = 'noisy_logits'
    else:
        noise_status = 'noiseless'
    print('lenght of inputs: ' , len(inputs))
    print('lenght of arr_models:', len(arr_models))
    for j in range(len(inputs)):
        arr_classification = []
        arr_distortion = [np.sum((adv[j:j+1]-inputs[j:j+1])**2)**.5 for adv in arr_adv]
        str_target = str(np.argmax(targets[j]))
        str_input_label = str(input_labels[j])
        if j%9 == 0:
            
            input_image_counter += 1
        
            file_output_summary_name ='testoutput_'+str(start+input_image_counter)+'_summary.txt'

            file_output_summary_dir = os.path.join(folder_image_save,
                                                   f"test_img_allmodel_{noise_status}_"+
                                                   str(start+input_image_counter))
        
            makeDir(file_output_summary_dir)
            print(f"file_output_summary_dir:{file_output_summary_dir}")
            file_output_summary = os.path.join(file_output_summary_dir,
                                               file_output_summary_name)
             
            f_summary = open(file_output_summary, 'w')
            f_summary.write('Begin Timestamp: '+str(timestart)+'\n')
            f_summary.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
        f_summary.write('Target: '+str_target+'\n')
        f_summary.write('Input: '+ str_input_label+ '\n')
        f_summary.write('all distortions: '+';'.join([str(dist) for dist in arr_distortion])+'\n')
    
        img_norm = np.sum((inputs[j:j+1])**2)**.5
        f_summary.write('Image norm: '+str(img_norm)+'\n')
        for ind in range(len(arr_models)):

            norm.append(img_norm)
            list_ind=[]
            list_ind.append(input_labels[j])
            list_ind.append(np.argmax(targets[j]))
            strImageAdv = arr_img_input[j]+'_adv'+str(ind)
            out_class_orig = [dict_class_orig[ind][k][j] for k in range(len(arr_models))]
            #print('out class orig',out_class_orig)
            for c in out_class_orig:
                list_ind.append(c)
            distort.append(arr_distortion[ind])
            out_class_adv = [dict_class_adv[ind][k][j] for k in range(len(arr_models))]
            for c in out_class_adv:
                list_ind.append(c)
            #print('counter: '+str(counter))
            #print('list_ind : '+str(list_ind))
            df.loc[counter]=list_ind
            counter+=1
            str_class_orig = ','.join([str(label) for label in out_class_orig])
            str_class_adv = ','.join([str(label) for label in out_class_adv])
            f_summary.write(strImageAdv+",original Classification: "+str_class_orig+'\n')
            f_summary.write(strImageAdv+",adv Classification: "+str_class_adv+'\n')
            f_summary.write(strImageAdv+',original vote: '+str(np.bincount(out_class_orig).argmax())+'\n')
            f_summary.write(strImageAdv+',adv vote: '+str(np.bincount(out_class_adv).argmax())+'\n')
        if j%9==8:
            #f_summary.write('End Timestamp: '+str(timeend)+'\n')
            f_summary.close()

    # print(len(norm))
    # print(len(distort))        
    df['image_norm']=norm
    df['distortion']=distort
    address0 = os.path.join(strTest,'attacked_single_models')
    address1 = os.path.join(address0,'attacked_single_networks_'+str(start)+'.txt')
    address2 = os.path.join(address0,'attacked_single_networks_'+str(start)+'.csv')
    makeDir(address0)
    df.to_csv(address1)
    df.to_csv(address2)

    end_time2 = time.time()
    print('time for final fors:', end_time-start_time)
    print("total time:",end_time2-start_time)
    attack_time = os.path.join(strTest,'attack_time.txt')
    s_summary = open(attack_time, 'a')
    s_summary.write('attack time:'+ str(end_time-timestart)+'\n') 
    s_summary.close()
    
    return df


# def get_prediction_array(model,data):
#     pred_model = model.predict(data)
#     preds = np.argmax(pred_model, axis = 1)
#     return list(preds)




def get_prediction_array(sess, model, x, feeds):
    pred_model = model.predict(x)
    preds = sess.run(pred_model, {x:feeds})
    return [np.argmax(pred) for pred in preds]


def get_superimposed(baseImage, arrAdvImage, tol=1e-10):
    arr_diff = []
    tmpImposed = baseImage
    for advImage in arrAdvImage:
        arr_diff.append(advImage-baseImage)
    for diff in arr_diff:
        tmpImposed = tmpImposed + diff
    tmpImposed[tmpImposed>0.5] = 0.5
    tmpImposed[tmpImposed<-0.5] = -0.5
    #print('shape of tmpImposed inside sup attack:',tmpImposed.shape)
    return np.squeeze(tmpImposed,axis=0)

