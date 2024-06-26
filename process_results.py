import numpy as np
import glob
import os
import pandas as pd
import shutil
def process_all_results(strFolder, bSupImp=False):
    g = os.path.join(strFolder,'*summary*.txt')
    arrFilenames = glob.glob(g)
    print('Mina Array file names:',arrFilenames)
    #he glob module is a useful part of the Python standard library. glob (short for global) is used to return all file paths that match a specific pattern
    f = open(os.path.join(strFolder,'master.txt'),'w')
    extract_method = extract_results
    if bSupImp:
        extract_method = extract_results_supimposed

    strHeader, arrResults = extract_method(arrFilenames[0])
    f.write(strHeader+'\n')
    for resLine in arrResults:
        f.write(resLine+'\n')
    for i in range(len(arrFilenames)-1):
        strHeader, arrResults = extract_method(arrFilenames[i+1])
        for resLine in arrResults:
            f.write(resLine+'\n')
    f.close()

def extract_results(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            if iTarget >= 9 and ('/' in line or 'End Timestamp' in line):
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}
    arrResults = []
    arrHeaders = ['input']
    for i, sample in dictSample2Target.items():
        strResult = ''
        if len(arrHeaders) < 2:
            tar = list(sample.keys())[0]
            for key in sample[tar].keys():
                if 'classification' in key:
                    arrHeaders.append(key)
            arrHeaders.append('final vote')
            if 'total distortion' in sample[tar].keys():
                arrHeaders.append('total distortion')
        for target, res in sample.items():
            strResult = strFilename+','+str(i)+','+str(target)+','+','.join([str(res[name]) for name in arrHeaders])
            arrResults.append(strResult)
    strHeader = 'filename,sample,target,'+','.join(arrHeaders)
    return strHeader, arrResults
            
def extract_results_supimposed(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        strDistortAll = ''
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                strKey = strKey.replace(os.path.sep,'')
                strKey = strKey.replace(' ','')
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            elif 'Image norm' in line:
                norm = float(line.split(': ')[1])
                dictTarget2Class[target]['image norm'] = norm
                dictTarget2Class[target]['all distortions'] = strDistortAll
            elif 'all distortions' in line:
                strDistortAll = line.split(': ')[1][:-1]
            if iTarget >= 9 and 'Image norm' in line:
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}    
    arrResults = []
    arrHeaders = ['input']
    for i, sample in dictSample2Target.items():
        strResult = ''
        if len(arrHeaders) < 2:
            tar = list(sample.keys())[0]
            for key in sample[tar].keys():
                if 'classification' in key:
                    arrHeaders.append(key)
            arrHeaders.append('final vote')
            if 'total distortion' in sample[tar].keys():
                arrHeaders.append('total distortion')
            if 'image norm' in sample[tar].keys():
                arrHeaders.append('image norm')
            if 'all distortions' in sample[tar].keys():
                arrHeaders.append('all distortions')
        for target, res in sample.items():
            strResult = strFilename+','+str(i)+','+str(target)+','+','.join([str(res[name]) for name in arrHeaders])
            arrResults.append(strResult)
    strHeader = 'filename,sample,target,'+','.join(arrHeaders)
    return strHeader, arrResults

def extract_results_supimposed_dict(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        strDistortAll = ''
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            elif 'Image norm' in line:
                norm = float(line.split(': ')[1])
                dictTarget2Class[target]['image norm'] = norm
                dictTarget2Class[target]['all distortions'] = strDistortAll
            elif 'all distortions' in line:
                strDistortAll = line.split(': ')[1][:-1]
            if iTarget >= 9 and 'Image norm' in line:
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}
    return dictSample2Target

def extract_stats(strMaster, arrTeachers ,strResult, num_attack):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_input = data['input']
    column_target = data['target']
    column_vote = data['final_vote']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    print('teacher,original accuracy,adv accuracy')
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'originalclassification'
        str_adv_class = strTeacher+'advclassification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        count_orig = sum(column_input[i]==column_teacher_orig[i] for i in range(nSamples))
        count_adv = sum(column_input[i]==column_teacher_adv[i] for i in range(nSamples))
        arr_orig_accuracy.append(count_orig)
        arr_adv_accuracy.append(count_adv)
        print(strTeacher+','+str(count_orig)+','+str(count_adv))
    print('student,output=input,output=target,output=other')
    arr_distortion = column_distortion/column_norm
    count_input = 0
    count_target = 0
    count_other = 0
    distort_input = 0.0
    distort_target = 0.0
    distort_other = 0.0
    for i in range(nSamples):
        if column_input[i]==column_vote[i]:
            count_input = count_input + 1
            distort_input = distort_input + arr_distortion[i]
        elif column_target[i]==column_vote[i]:
            count_target = count_target + 1
            distort_target = distort_target + arr_distortion[i]
        else:
            count_other = count_other + 1
            distort_other = distort_other + arr_distortion[i]
    if count_input > 0: distort_input = distort_input/count_input
    if count_target > 0: distort_target = distort_target/count_target
    if count_other > 0: distort_other = distort_other/count_other
    print('count,: unchanged: '+str(count_input)+', To target: '+str(count_target)+',To others: '+str(count_other))
    print('\n'+'average distortion :'+str(distort_input)+','+str(distort_target)+','+str(distort_other))
    b=open(strResult,'w')
    
    b.write('count,'+str(count_input)+','+str(count_target)+','+str(count_other)+'\n'+
            'unchanged accuracy: '+str(count_input/num_attack)+ '\n'+'corret to target accuracy: '+str(count_target/num_attack)+
            '\n'+'corret to other accuracy: '+str(count_other/num_attack)+'\n')
    b.write('average distortion:'+str(distort_input)+','+str(distort_target)+','+str(distort_other))
    
    b.close()
    return count_input/num_attack , count_target/num_attack , count_other/num_attack , distort_input , distort_target , distort_other


def extract_details(strMaster, arrTeachers):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_input = data['input']
    column_target = data['target']
    column_vote = data['final_vote']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    arr_teacher_orig = []
    arr_teacher_adv = []
    arr_correct = []
    arr_target = []
    arr_other = []
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'_original_classification'
        str_adv_class = strTeacher+'_adv_classification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        arr_teacher_orig.append(column_teacher_orig)
        arr_teacher_adv.append(column_teacher_adv)
    arr_distortion = column_distortion/column_norm
    print(',output=input,output=target,output=other')
    for i in range(nSamples): 
        count_input = 0
        count_target = 0
        count_other = 0
        distort_input = 0.0
        distort_target = 0.0
        distort_other = 0.0
        for j in range(len(arrTeachers)):
            if column_input[i]==arr_teacher_adv[j][i]:
                count_input = count_input + 1
            elif column_target[i]==arr_teacher_adv[j][i]:
                count_target = count_target + 1
            else:
                count_other = count_other + 1
        arr_correct.append(count_input)
        arr_target.append(count_target)
        arr_other.append(count_other)
        print(str(i)+','+str(count_input)+','+str(count_target)+','+str(count_other))


def get_final_vote(strMaster, arrTeachers,strFileout=''):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_sample = data['sample']
    column_input = data['input']
    column_target = data['target']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    column_alldistort  = data['all_distortions']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    arr_teacher_orig = []
    arr_teacher_adv = []
    arr_correct = []
    arr_target = []
    arr_other = []
    f_out = None
    if strFileout != '':
        f_out = open(strFileout,'w')
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'_original_classification'
        str_adv_class = strTeacher+'_adv_classification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        arr_teacher_orig.append(column_teacher_orig)
        arr_teacher_adv.append(column_teacher_adv)
    arr_distortion = column_distortion/column_norm
    strheader = 'sample,target,input,original vote,noisy vote,total_distortion,image_norm,all distortions'
    if f_out != None:
        f_out.write(strheader+'\n')
    print(strheader)
    for i in range(nSamples):
        arr_orig = []
        arr_adv = []
        for j in range(len(arrTeachers)):
            arr_orig.append(arr_teacher_orig[j][i])
            arr_adv.append(arr_teacher_adv[j][i])
        str_orig = str(np.bincount(arr_orig).argmax())
        str_adv = str(np.bincount(arr_adv).argmax())
        str_out = str(column_sample[i])+','+str(column_target[i])+','+str(column_input[i])
        str_out = str_out+','+str_orig+','+str_adv+','+str(column_distortion[i])+','+str(column_norm[i])+','+str(column_alldistort[i])
        print(str_out)
        if f_out != None:
            f_out.write(str_out+'\n')
    if f_out != None:
        f_out.close()
    

def get_transfer_stats(strfilename,strFileout=''):
    fIn = open(strfilename,'r')
    strheader = 'Input,Target,Adv,Distortion,Imagen_norm,correct_to_target,correct_to_other,orig_vote,adv_vote'
    strtarget = ''
    strinput = ''
    strtarget = ''
    stradv = ''
    strdistortion = ''
    strimagenorm = ''
    correct2target = 0
    correct2other = 0
    strvote_orig = ''
    strvote_adv = ''
    arr_target = [str(i) for i in range(10)]
    arr_out = []
    for line in fIn:
        if 'Target: ' in line:
            if strtarget != '':
                strtarget = ''
                strinput = ''
                strtarget = ''
                stradv = ''
                strdistortion = ''
                strimagenorm = ''
                correct2target = 0
                correct2other = 0
                strvote_orig = ''
                strvote_adv = ''
            strtarget = line[len('Target: '):].strip('\n')
            arr_target.remove(strtarget)
        elif 'all distortions: ' in line:
            arrdistort_all = line[len('all distortions: '):].strip('\n').split(';')
        elif 'Image norm: ' in line:
            strimagenorm = line[len('Image norm: '):].strip('\n')
        elif 'Classification:' in line:
            if 'original ' in line:
                strline_orig = line.strip('\n')
            else:
                strline_adv = line.strip('\n')
                stradv, correct2target, correct2other = get_line_stats(strtarget,strline_orig,strline_adv)
                strdistortion = arrdistort_all[int(stradv)]
        elif 'vote: ' in line:
            if 'original ' in line:
                itmp = line.find('original vote: ')+len('original vote: ')
                strvote_orig = line[itmp:].strip('\n')
            else:
                itmp = line.find('adv vote: ')+len('adv vote: ')
                strvote_adv = line[itmp:].strip('\n')
                strout = strtarget+','+stradv+','+strdistortion+','+strimagenorm+','+str(correct2target)+','+str(correct2other)+','+strvote_orig+','+strvote_adv
                print(strout)
                arr_out.append(strout)
    if (strFileout != ''):
        fOut = open(strFileout,'w')
        fOut.write(strheader+'\n')
        strinput = arr_target[0]
        for strout in arr_out:
            fOut.write(strinput+','+strout+'\n')
        fOut.close()
        c = os.path.join(strFileout)

        

def get_line_stats(strtarget,strline_orig,strline_adv):
    arrsplit_orig = strline_orig.split(',')
    arrsplit_adv = strline_adv.split(',')
    iadv = arrsplit_orig[0].find('target'+strtarget)+len('target'+strtarget)
    stradv = arrsplit_orig[0][iadv+4:]
    itmp = arrsplit_orig[1].find(': ')
    arrsplit_orig[1] = arrsplit_orig[1][itmp+2:]
    iadv = arrsplit_adv[0].find('target'+strtarget)+len('target'+strtarget)
    itmp = arrsplit_adv[1].find(': ')
    arrsplit_adv[1] = arrsplit_adv[1][itmp+2:]
    count2target = 0
    count2other = 0 
    for i in range(1,len(arrsplit_orig)):
        if arrsplit_adv[i] != arrsplit_orig[i]:
            if arrsplit_adv[i] == strtarget:
                count2target = count2target+1
            else:
                count2other = count2other+1
    return stradv, count2target, count2other


def single_process(strFolder1,strResult,treshhold,num_adv_exmples,noisy_logit):
    if noisy_logit == True:
         strFolder = os.path.join(strFolder1,'single_attack_noisy')
    elif noisy_logit == False:
         strFolder = os.path.join(strFolder1,'single_attack_noiseless')
    distortion_unchanged = 0
    distortion_c2t = 0
    distortion_c2o = 0         
    correct_to_target = 0
    correct_to_other = 0
    unchanged = 0
    orig_correct = 0
    more_than_treshhold = 0
    g = os.path.join(strFolder,'*stats*.txt')
    arrFilenames = glob.glob(g)
    counter1=0
    #print(arrFilenames)

    for stats in arrFilenames:
        with open(stats) as f:
            data = pd.read_csv(stats)
            overal_file = os.path.join(strFolder,'overal.txt')
            df = pd.DataFrame(data)
            if counter1==0:
                df.to_csv(overal_file, header = True, index = False , mode='w')
            else:
                df.to_csv(overal_file, header = False, index = False , mode='a')
            print(counter1)
                
            counter1+=1
            #Correctly classified before attack images
            orig_correct = orig_correct +sum((df['Input']==df['orig_vote']))
            df0 = df[(df['Input'] == df['adv_vote'])].copy()
            df0_1 =df0[(df0['Distortion'] <= treshhold*df0['Imagen_norm'])]
            df0_1['relative_distortion']= df0_1['Distortion']/df0_1['Imagen_norm']
            distortion_unchanged+=sum(df0_1['relative_distortion'])
            #Correclty classified after attack images
            unchanged = unchanged + sum((df['Input'] == df['adv_vote']))
            df1 = df[(df['Input'] != df['adv_vote'])].copy()
            # df1['failed'] = np.where(df1['Distortion'] <= (treshhold*df1['Imagen_norm']), 1, 0)
            # failed_attacks = sum(df1['failed'])
            # print('failed attacks',df1['failed'])
            #I need to eliminate images with high distortion
            more_than_treshhold = more_than_treshhold + sum(df1['Distortion']>(treshhold*df1['Imagen_norm']))
            df2 = df1[(df1['Distortion']<=(treshhold*df1['Imagen_norm']))].copy()
            df2['relative_distortion']= df2['Distortion']/df2['Imagen_norm']
            #Images classified as the target of the attack
            df3 = df2[(df2["Target"]==df2['adv_vote'])]
            distortion_c2t+=sum(df3['relative_distortion'])
            correct_to_other =correct_to_other+ sum((df2['Target'] != df2['adv_vote']))
            #Images classified as neither input class nor the target class of the attack
            df4 = df2[(df2["Target"]!=df2['adv_vote'])]
            distortion_c2o+=sum(df4['relative_distortion'])
            correct_to_target =correct_to_target+ sum((df2['Target'] == df2['adv_vote']))

    print('final results: Unchanged: ', unchanged ,
          'successfull attacks with perturbation more than limit: ', more_than_treshhold ,
          'Correct to target: ', correct_to_target ,
          'Correct to other: ' , correct_to_other)
    if treshhold == 0.75:
        b=open(strResult,'w')
    else:
        b=open(strResult,'a')
    if correct_to_target == 0:
        c2t = 0
    else:
        c2t = distortion_c2t/correct_to_target
    if correct_to_other == 0:
        c2o = 0
    else:
        c2o = distortion_c2o/correct_to_other
    if unchanged == 0:
        unch = 0
    else:
        unch = distortion_unchanged/unchanged
    b.write( '          _______________________________________________________         '+
            '\n' + 'treshhold: ' + str(treshhold)+
            '\n'+ 'Ensemble  statistics: '+
            '\n'+ 'number of adversarial examples: '+str(num_adv_exmples)+
            '\n'+ 'correct original vote: '+str(orig_correct)+
            '\n'+ 'accuracy on original images: '+str(orig_correct/(num_adv_exmples))+
            '\n'+ 'final results: Unchanged: '+str(unchanged)+
            '\n'+ 'perturbation more than limit: '+ str(more_than_treshhold) +
            '\n'+ 'Correct to target: '+str(correct_to_target)+'  Correct to other: '+str(correct_to_other)+
            '\n' + 'accuracy(without considering perturbation limit): ' + str(unchanged/num_adv_exmples)+
            '\n' + 'accuracy(after removing examples with perturbation higher than treshhold): ' +
            str((unchanged)/(num_adv_exmples-more_than_treshhold)) +
            '\n'+ 'perturbation for unchanged:'+str(distortion_unchanged)+
            '\n'+ 'perturbation for correct to target: '+str(distortion_c2t)+
            '\n'+ 'perturbation for correct to other: '+str(distortion_c2o)+
            '\n'+ 'average perturbation for unchanged: '+str(unch)+
            '\n'+ 'average perturbation for correct to target: '+str(c2t)+
            '\n'+ 'average perturbation for correct to other: '+str(c2o))
    
    b.close()
    accuracy_after_attack = (unchanged)/(num_adv_exmples-more_than_treshhold)
    c_to_t = correct_to_target/(num_adv_exmples-more_than_treshhold)
    c_to_o = correct_to_other/(num_adv_exmples-more_than_treshhold)
    return  accuracy_after_attack , c_to_t , c_to_o ,  unch , c2t , c2o , (orig_correct/(num_adv_exmples))

def individual_statistics(df_address,num_teachers,num_adv_exmples,noisy_logit):
    add0 = os.path.join(df_address,'attacked_single_models')
    k = os.path.join(add0,'*single_networks*.txt')
    add1 = os.path.join(df_address,'attacked_single_models_total.txt')
    arrFilenames = glob.glob(k)
    
    for ii,files in enumerate(arrFilenames):     
        df = pd.read_csv(files)
        if ii == 0:
            df.to_csv(add1, header = True, index = True , mode='w')
        else:
            df.to_csv(add1, header = False , index = True , mode='a')
    df_total = pd.read_csv(add1) 
    grey_images = sum((df_total['distortion']==df_total['image_norm']))
    df0 = df_total[(df_total['distortion'] < df_total['image_norm'])]
    len_df = df0.shape[0]
    ind_orig_correct = [0 for i in range(num_teachers)]
    ind_correct_to_target = [0 for i in range(num_teachers)]
    ind_correct_to_other = [0 for i in range(num_teachers)]
    ind_unchanged = [0 for i in range(num_teachers)]
    for i in range(num_teachers):        
        stradv= 'adv_classification_'+str(i)
        strorig = 'original_classification_'+str(i)
        ind_orig_correct[i] = sum(df0['input_label']==df0[strorig])
        ind_unchanged[i] = sum(df0['input_label']==df0[stradv])
        ind_correct_to_target[i] = sum(df0['Target']==df0[stradv])
        df1 = df0[(df0['input_label']!=df0[stradv])]
        ind_correct_to_other[i] = sum(df1['Target']!=df1[stradv])
    print('length of examples:'+str(len_df))

    b = open(add1,'a')
    b.write('\n'+'    ____________________________________    '+
            '\n'+'Individuals statistics: '+
            '\n'+'Original correct: ' + ','.join([str(x) for x in ind_orig_correct])+
            '\n'+'unchanged: '+','.join([str(y) for y in ind_unchanged]) +
            '\n'+'Correct to target: '+','.join([str(z) for z in ind_correct_to_target]) +
            '\n'+'Correct to other: '+ ','.join([str(v) for v in ind_correct_to_other])+
            '\n' +'original accuracy for each network:' + 
            ','.join([str(ind_orig_correct[i]/len_df)+' , ' for i in range(num_teachers)])+
            '\n' +'mean of original accuracies for networks :'+ str(np.mean(ind_orig_correct))+
            '\n' +'std of original accuracies for networks :'+ str(np.std(ind_orig_correct))+           
            '\n'+'accuracy after attack for each network:' + 
            ','.join([str(ind_unchanged[i]/len_df)+' , ' for i in range(num_teachers)])+
            '\n'+'mean of accuracies after attack networks:' + str(np.mean(ind_unchanged)) +
            '\n'+'std of accuracies after attack networks:' + str(np.std(ind_unchanged)) +            
            '\n'+'number of grey images: '+str(grey_images)+
            '\n'+'average original accuracies: '+str((sum(ind_orig_correct)/len_df)*100/num_teachers)+ '%' +
            '\n'+'average accuracies after attack:'+str((sum(ind_unchanged)/len_df)*100/num_teachers)+ '%')
    b.close
 

def get_min_max_details(strfile,output):
    table = np.genfromtxt(strfile,delimiter=',',names=True)
    input_col = table['Input']
    target_col = table['Target']
    adv_vote_col = table['adv_vote']
    orig_vote_col = table['orig_vote']
    correct2target_col = table['correct_to_target']
    correct2other_col = table['correct_to_other']
    distort_col = table['Distortion']
    orig_input_count = len([1 for i in range(len(input_col)) if input_col[i]==orig_vote_col[i]])
    adv_input_count = len([1 for i in range(len(input_col)) if input_col[i]==adv_vote_col[i]])
    adv_target_count = len([1 for i in range(len(target_col)) if target_col[i]==adv_vote_col[i]])
    avg_correct2target = np.mean(correct2target_col)
    avg_correct2other = np.mean(correct2other_col)
    min_distort = []
    max_distort = []
    min_correct2target = []
    min_correct2other = []
    max_correct2target = []
    max_correct2other = []
    arr_target = list(set(target_col))
    for target in arr_target:
        tmp_table = table[(table['Target']==target)]
        tmp_ind = np.array(tmp_table['Distortion']).argsort()
        min_distort.append(tmp_table['Distortion'][tmp_ind[0]])
        max_distort.append(tmp_table['Distortion'][tmp_ind[-1]])
        min_correct2target.append(tmp_table['correct_to_target'][tmp_ind[0]])
        max_correct2target.append(tmp_table['correct_to_target'][tmp_ind[-1]])
        min_correct2other.append(tmp_table['correct_to_other'][tmp_ind[0]])
        max_correct2other.append(tmp_table['correct_to_other'][tmp_ind[-1]])
    print(strfile)
    print('correct lable on orig input: ', orig_input_count)
    print('correct label on adv input: ', adv_input_count)
    print('target label on adv input: ', adv_target_count)
    print('avg correct flipped to target: ', avg_correct2target)
    print('avg correct flipped to other: ', avg_correct2other)
    print('min distortion: ', min_distort)
    print('max distortion: ', max_distort)
    print('min distortion, label flipped to target: ',min_correct2target)
    print('min distortion, label flipped to other: ',min_correct2other)
    print('max distortion, label flipped to target: ',max_correct2target)
    print('max distortion, label flipped to other: ',max_correct2other)
    address = os.path.join(output,'min_max_details.txt')
    d= open(address,'w')
    d.write('correct lable on orig input: '+str(orig_input_count)+ '\n'
    +'correct label on adv input: '+ str(adv_input_count)+ '\n'
    +'target label on adv input: '+ str(adv_target_count)+ '\n'
    +'avg correct flipped to target: '+ str(avg_correct2target)+ '\n'
    +'avg correct flipped to other: '+ str(avg_correct2other)+ '\n'
    +'min distortion: '+ str(min_distort)+ '\n'
    +'max distortion: '+str(max_distort)+ '\n'
    +'min distortion, label flipped to target: '+str(min_correct2target)+ '\n'
    +'min distortion, label flipped to other: '+str(min_correct2other)+ '\n'
    +'max distortion, label flipped to target: '+str(max_correct2target)+ '\n'
    +'max distortion, label flipped to other: '+str(max_correct2other)+ '\n')

#single_process('C:/Users/miyaz/Documents/GitHub/Yuting_code_modifications/Result/CIFAR_diverse_1_150individual_epochs_5teachers','C:/Users/miyaz/Documents/GitHub/Yuting_code_modifications/Result/CIFAR_diverse_1_150individual_epochs_5teachers/single_attack/final_result.txt')