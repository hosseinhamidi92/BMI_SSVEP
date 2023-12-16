from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
from Data_Loader import Data_preparing
from Models import *
from pathlib import Path
import os
from scipy.signal import hilbert
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score,accuracy_score
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import time
def train(train_data, labels, test_data, labels_test ,model_PARAMS,sub, sub_name ,model_name, num_folds=10, save= False):
    '''
    

    Parameters
    ----------
    train_data : np.array
        train signal.
    labels : np.array
        train lebels.
    test_data : np.array
        test signa.
    labels_test : np.array
        test lebels.
    model_PARAMS : dict
        Model Parameters.
    sub : int
        suject (or experiment) number.
    sub_name : str
        subject name.
    model_name : fun
        name of model for training.
    num_folds : int, optional
        cross validation. The default is 10.
    save : bool, optional
        whether save the model or not. The default is False.

    Returns
    -------
    int
        average of cross fold accuracy and the max of test acc.

    '''
    
    subject = sub_name + '_%d.h5'%sub
    #subject = 'RR1_ Pretrain %d.h5'%sub
    kf = KFold(n_splits=num_folds, shuffle=False)
    kf.get_n_splits(train_data)
    cv_acc = np.zeros((num_folds, 1))
    fold = -1
    idd_acc = 0
    tst_acc = []
    for train_index, test_index in kf.split(train_data):
        x_tr, x_ts = train_data[train_index], train_data[test_index]
        y_tr, y_ts = labels[train_index], labels[test_index]
        input_shape = np.array([x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]])
        
        fold = fold + 1
        
        
        #model = CNN_model(input_shape, CNN_PARAMS)
        #model = EEGNet(input_shape, CNN_PARAMS)
        model = model_name(input_shape, model_PARAMS)
        #model.summary()

        adm = optimizers.Adam(lr=model_PARAMS['learning_rate'])
        #sgd = optimizers.SGD(lr=model_PARAMS['learning_rate'], decay=model_PARAMS['lr_decay'], momentum=model_PARAMS['momentum'], nesterov=False)
        model.compile(loss=categorical_crossentropy, optimizer=adm, metrics=["accuracy"])
        tr_time1 =  time.time()
        history = model.fit(x_tr, y_tr, batch_size=model_PARAMS['batch_size'], 
                            epochs=model_PARAMS['epochs'], verbose=0)
        tr_time2 =  time.time()
        print('Training time: %f'%(tr_time2 - tr_time1))
        
        score = model.evaluate(x_ts, y_ts, verbose=0) 
        cv_acc[fold, :] = score[1]*100
        
        if score[1]*100>idd_acc:
            if save:
                model.save(subject)
            idd_acc = score[1]*100
            #print('idd_acc:',idd_acc)
            te_time1 =  time.time()
            test_loss, test_acc = model.evaluate(test_data, labels_test)
            te_time2 =  time.time()
            print('Inference time: %f'%((te_time2 - te_time1)/len(labels_test)))
            #print('test_acc:',test_acc)
            tst_acc.append(test_acc)
            
        print(f'cv{fold+1}:{score[1]*100:.2f}%', end=" ")
    #print('mean acc=', np.mean(cv_acc))
    return np.mean(cv_acc), np.max(tst_acc)

def test( dte, label_te, model_dir, img_dir):
    model = load_model(model_dir)
    ypred = model.predict(dte)
    # dot_img_file = '/model_1.png'
    # plot_model(model, to_file=dot_img_file, show_shapes=False)

    f1 = f1_score(label_te, np.round(ypred), average=None)
    acc = accuracy_score(label_te, np.round(ypred))
    print(f'F1 score {f1}, Accuracy{acc*100:.2f}%', end=" ")
    # -2, -13, -17
    layers = [-2,-13,-17]
    for layer in layers:
        intermediate_layer_model = Model(inputs=model.input,
                                           outputs=model.layers[layer].output)
        intermediate_output = intermediate_layer_model.predict(dte)
        intermediate_output = intermediate_output.reshape((dte.shape[0],-1))
        intermediate_output = intermediate_output.reshape((-1, intermediate_output.shape[1]))
        color = label_te
        color = [np.argmax(i) for i in color] # take one-hot Convert encoding to integer 
        color = np.stack(color, axis=0)
        group = color
        cdict = {0:'black', 1: 'red', 2: 'blue', 3: 'green'}
    
        n_neighbors = 4   # How many categories are there 
    
        ##### 2D feature plot
        y = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(intermediate_output)
        scatter_x = y[:, 0]
        scatter_y = y[:, 1]
        fig = plt.figure(figsize=(12, 12))
        for g in np.unique(group):
            ix = np.where(group == g)
            plt.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 10)
       # plt.legend()
        plt.axis('off')
        #plt.show()
        #plt.title('t-SNE Scatter Plot', fontsize=14)
        layer_name = '%d'%layer
        plt.savefig(img_dir[:-5]+'_Layer'+layer_name+'_2D.png')

    ##### 3D feature plot
    # y = TSNE(n_components=3, learning_rate='auto',init='pca').fit_transform(intermediate_output)
    # scatter_x = y[:, 0]
    # scatter_y = y[:, 1]
    # scatter_z = y[:, 2]
    
    # fig = plt.figure(figsize=(12, 12))
    # #fig, ax = plt.subplots()
    # ax = fig.add_subplot(projection='3d')
    # for g in np.unique(group):
    #     ix = np.where(group == g)
    #     ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c = cdict[g], label = g, s = 100)
    # ax.legend()
    # plt.title('t-SNE Scatter Plot', fontsize=14)
    # plt.savefig(img_dir)
    
def read_and_train(windows_lengths, FFT_PARAMS,model_PARAMS, directory = 'all_data_exp_3_only_filter_applied/'):
    '''
    

    Parameters
    ----------
    windows_lengths : list
        list of the window lengths.
    FFT_PARAMS : dict
        FFT parameters.
    model_PARAMS : dict
        Model parameters.
    directory : str, optional
        The directory of data (only data should be available in that folder). The default is 'all data exp_3 (only filter applied)/'.

    Returns
    -------
    Accs_time_slots : dict
        A dictionary with the cross val cc for each subject in different window length.

    '''
    
    files = os.listdir(directory)
    Accs_time_slots = dict()
    for subject in files:
        sub_acc = []
        sub_name = subject[0:-4]
        sub_dir = directory + subject
        for window_len in windows_lengths:
            d_pre = Data_preparing(sub_dir,FFT_PARAMS,500, FFT=True, window_len = window_len, trial_ratio = 44 )
            dt, label, dte,label_te = d_pre.concat_subs(window_len, 0.25)
            mean_acc_cv, _ = train(dt, label, dte, label_te,model_PARAMS ,0, sub_name ,load_Model2, num_folds=10, save= False)
            sub_acc.append(mean_acc_cv)
        Accs_time_slots[sub_name] = sub_acc
        print(sub_name)
    return Accs_time_slots



def shuffle(dt,label):
    rng = np.random.default_rng()
    ind = np.arange(dt.shape[0])
    rng.shuffle(ind)
    dt = dt[ind]
    label = label[ind]
    return dt,label
    


def read_and_train_on_all_timeslots(windows_lengths, FFT_PARAMS,model_PARAMS, directory = 'all_data_exp_3_only_filter_applied/'):
    '''
    

    Parameters
    ----------
    windows_lengths : list
        list of the window lengths.
    FFT_PARAMS : dict
        FFT parameters.
    model_PARAMS : dict
        Model parameters.
    directory : str, optional
        The directory of data (only data should be available in that folder). The default is 'all data exp_3 (only filter applied)/'.

    Returns
    -------
    Accs_time_slots : dict
        A dictionary with the cross val cc for each subject in different window length.

    '''
    
    files = os.listdir(directory)
    Accs_time_slots = dict()
    for subject in files:
        sub_acc = []
        sub_name = subject[0:-4]
        sub_dir = directory + subject
        train_inp = []
        test_inp = []
        train_label = []
        test_label = []
        # for window_len in windows_lengths:
        #     d_pre = Data_preparing(sub_dir,FFT_PARAMS,500, FFT=True, window_len = window_len, trial_ratio = 30 )
        #     dt, label, dte,label_te = d_pre.concat_subs(window_len, window_len)
        #     #dt = dt[:,5:14,:,:]
        #     #dte = dte[:,5:14,:,:]
        #     train_inp+=list(dt)
        #     train_label+=list(label)
        #     test_inp+=list(dte)
        #     test_label+=list(label_te)
        
        for window_len in windows_lengths:
            d_pre = Data_preparing(sub_dir,FFT_PARAMS,500, FFT=True, window_len = window_len, trial_ratio = 30 )
            dt, label, dte,label_te = d_pre.concat_subs(0.05*window_len, window_len)


            #d_pre = Data_preparing(sub_dir,FFT_PARAMS,500, FFT=True, window_len = window_len, trial_ratio = 30 )
            #dt, label, dte,label_te = d_pre.concat_subs(window_len, window_len)
            #channel_list = [11,13,14]
            #dt = dt[:,channel_list,:,:]
            #dte = dte[:,channel_list,:,:]
            
            # dt = hilbert(dt)
            # dt = np.abs(dt)
            
            # dte = hilbert(dte)
            # dte = np.abs(dte)
            train_inp+=list(dt)
            train_label+=list(label)
            test_inp+=list(dte)
            test_label+=list(label_te)
        dt = np.array(train_inp)
        label = np.array(train_label)
        
        dt,label = shuffle(dt,label)
        mean_acc_cv, _ = train(dt, label, np.array(test_inp), np.array(test_label),model_PARAMS ,0, sub_name ,CNN_model, num_folds=10, save= True)
        #mean_acc_cv, _ = train(dt, label, np.array(test_inp), np.array(test_label),model_PARAMS ,0, sub_name ,load_Model2, num_folds=10, save= True)
        sub_acc.append(mean_acc_cv)
        Accs_time_slots[sub_name] = sub_acc
        print(sub_name)
    return Accs_time_slots


def test_on_all_timeslots(windows_lengths, FFT_PARAMS,model_dir = 'Train_on_all_timeslots_shuffled_overlap(0.05)_res 0.3_30 trial/', directory = 'all_data_exp_3_only_filter_applied/'):
    '''
    

    Parameters
    ----------
    windows_lengths : list
        list of the window lengths.
    FFT_PARAMS : dict
        FFT parameters.
    model_dir : str
        saved Model directory.
    directory : str, optional
        The directory of data (only data should be available in that folder). The default is 'all data exp_3 (only filter applied)/'.

    Returns
    -------
    Accs_time_slots : dict
        A dictionary with the cross val cc for each subject in different window length.

    '''
    
    files = os.listdir(directory)
    Accs_time_slots = dict()
    for subject in files:
        sub_acc = []
        sub_name = subject[0:-4]
        sub_dir = directory + subject
        sub_model_dir = model_dir + sub_name+'.h5'
        img_dir = model_dir + sub_name+'.png'
        train_inp = []
        test_inp = []
        train_label = []
        test_label = []
        
        for window_len in windows_lengths:
            d_pre = Data_preparing(sub_dir,FFT_PARAMS,500, FFT=True, window_len = window_len, trial_ratio = 30 )
            dt, label, dte,label_te = d_pre.concat_subs(window_len*0.05, window_len*0.05)
            #dt, label, dte,label_te = d_pre.concat_subs(window_len, window_len)
            #channel_list = [11,13,14]
            #dt = dt[:,channel_list,:,:]
            #dte = dte[:,channel_list,:,:]
            # dt = hilbert(dt)
            # dt = np.abs(dt)
            
            # dte = hilbert(dte)
            # dte = np.abs(dte)
            train_inp+=list(dt)
            train_label+=list(label)
            test_inp+=list(dte)
            test_label+=list(label_te)
        dte = np.array(test_inp)
        label_te = np.array(test_label)
        
        dte,label_te = shuffle(dte,label_te)
        test( dte, label_te,sub_model_dir, img_dir)
        #mean_acc_cv, _ = train(dt, label, np.array(test_inp), np.array(test_label),model_PARAMS ,0, sub_name ,load_Model2, num_folds=10, save= True)
        #sub_acc.append(mean_acc_cv)
        #Accs_time_slots[sub_name] = sub_acc
        print(sub_name)
    return Accs_time_slots


if __name__ == "__main__":
    

    FFT_PARAMS = {
        'resolution': 0.2930,
        #'resolution': 0.1,
        'start_frequency': 3.0,
        'end_frequency': 35.0,
        'sampling_rate': 500
    }
    
    model_PARAMS = {
        'batch_size': 32,
        'epochs': 50,
        'droprate': 0.25,
        'learning_rate': 0.001,
        'lr_decay': 0.0,
        'l2_lambda': 0.0001,
        'momentum': 0.9,
        'kernel_f': 10,
        'n_ch': 16,
        'num_classes': 4}
    windows_lengths = [0.25,0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    #windows_lengths = [1]
    #test_on_all_timeslots(windows_lengths, FFT_PARAMS)
    Accs = read_and_train_on_all_timeslots(windows_lengths, FFT_PARAMS,model_PARAMS, directory = 'all_data_exp_3_only_filter_applied/')
    #Accs = read_and_train(windows_lengths, FFT_PARAMS,model_PARAMS, directory = 'all_data_exp_3_only_filter_applied/')


