import numpy as np

from numpy import matlib as mb
from tensorflow.keras.utils import to_categorical
from ripser import ripser
from scipy import sparse
import math

class Data_preparing():
    def __init__(self,directory,FFT_PARAMS,sample_rate, FFT=True, Purpose='Pretrain', pers = False, window_len = 1, trial_ratio = 30):
        '''
        Parameters
        ----------
        directory : str
            The directory of the dataset.
        FFT_PARAMS : dict
            Parameters required for FFT.
        sample_rate : int
            Data sample rate.
        FFT : Bool, optional
            Whether you want to apply FFT on data or not. The default is True.
        Purpose : str, optional
            {'Pretrain' or 'Target'}. The default is 'Pretrain'.
        pers : Bool, optional
            Obtain Persistent diagram or not. The default is False.
        window_len : int, optional
            The time in which you want to split data (second). The default is 1.
        trial_ratio : int, optional
            Number of trial that you want to use for training (the rest will be used for test). The default is 30.

        Returns
        -------
        None.

        '''
        
        
        self.fft_par = FFT_PARAMS
        self.sample_rate = sample_rate
        self.FFT = FFT
        self.Purpose = Purpose
        self.pers = pers
        self.window_len = window_len
        #self.shift_len = shift_len
        data = np.load(directory)
        #test_ratio = int(2* data.shape[-1] //3)
        self.init_train = data[:,:,:,:,:trial_ratio]
        self.init_test = data[:,:,:,:,trial_ratio:]
    
    
    
    
    def data_conc(self,dt):
        '''
        

        Parameters
        ----------
        dt : np.array
            Input signal.

        Returns
        -------
        dt_d : np.array
            Concatenated labels.

        '''
        dt_dd = dt[:,:,0,:]
        dt_dd1 = dt[:,:,1,:]
        dt_dd2 = dt[:,:,2,:]
        dt_dd3 = dt[:,:,3,:]

        dt_d = np.concatenate((dt_dd, dt_dd1 , dt_dd2, dt_dd3),axis = -1)
        dt_d = np.moveaxis(dt_d, [0,1,2], [1,2,0])
        return dt_d




    def Data_split(self,dt):
        '''
        

        Parameters
        ----------
        dt : np.array
            Input signal.

        Returns
        -------
        Input : np.array
            Splitted signals.
        Target : np.array
            label for each splitted signal.

        '''

        target = np.zeros((dt.shape[0],1))
        target[int(dt.shape[0]/4):2*int(dt.shape[0]/4)] = 1
        target[2*int(dt.shape[0]/4):3*int(dt.shape[0]/4)] = 2
        target[3*int(dt.shape[0]/4):4*int(dt.shape[0]/4)] = 3

        ind = np.random.permutation(len(dt))

        dt = dt[ind]
        target = target[ind]

        split_rate = int(dt.shape[-1]/(self.sample_rate * self.window_len))
        dt = dt[:,:,: int(split_rate * self.sample_rate * self.window_len)]

        a = np.split(dt, split_rate, axis = -1)
        aa = []
        for i in range(len(a)):
            for j in range(len(a[i])):
                aa.append(a[i][j])  

        t1 = target
        for i in range(split_rate-1):
            target = np.concatenate((target, t1))

        aa = np.array(aa)
        ind = np.random.permutation(len(aa))

        Input = aa
        #Input = aa[ind]
        Input = Input.reshape(-1,Input.shape[1],Input.shape[2],1)
        Target = target
        #Target = target[ind]
        
        return Input, Target
    
    def complex_spectrum_features(self,segmented_data):
        '''
        

        Parameters
        ----------
        segmented_data : np.array
            Input signal.

        Returns
        -------
        features_data : np.array
            FFT signal.

        '''
        
        num_trials = segmented_data.shape[0]
        num_chan = segmented_data.shape[1]


        fft_len = segmented_data[0, 0, :,0].shape[0]

        NFFT = round(self.fft_par['sampling_rate']/self.fft_par['resolution'])
        fft_index_start = int(round(self.fft_par['start_frequency']/self.fft_par['resolution']))
        fft_index_end = int(round(self.fft_par['end_frequency']/self.fft_par['resolution']))+1

        features_data = np.zeros((segmented_data.shape[0],segmented_data.shape[1], 2*(fft_index_end - fft_index_start), 
                                  segmented_data.shape[3]))

        for trial in range(0, num_trials):
            for channel in range(0, num_chan):
                temp_FFT = np.fft.fft(segmented_data[trial, channel, :, 0], NFFT)/fft_len
                real_part = np.real(temp_FFT)
                imag_part = np.imag(temp_FFT)
                features_data[trial,channel,:, 0] = np.concatenate((
                    real_part[fft_index_start:fft_index_end,], 
                    imag_part[fft_index_start:fft_index_end,]), axis=0)

        return features_data
    
    
    
    from ripser import ripser
    from scipy import sparse
    def pers(self,train_data):
        '''
        

        Parameters
        ----------
        train_data : np.array.
            Input time series data

        Returns
        -------
        dgm0 : np.array
            Persistent diagram of the signal.

        '''
        
        
        N = 500
        I = np.arange(N-1)
        J = np.arange(1, N)
        V = np.maximum(train_data[0:-1], train_data[1::])
        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, train_data))
        #Create the sparse distance matrix
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        dgm0 = ripser(D, maxdim=1, distance_matrix=True)['dgms'][0]
        return dgm0

    def convert_pers_to_matrix(self,train_data):           
        b = np.zeros((train_data.shape[0],16,60,2))

        for j in range(train_data.shape[0]):
            for ii in range(16):
                dgm0 = self.pers(train_data[j,ii,:,0])[:-1]
                dgm0[:,0] = (dgm0[:,0] - dgm0[:,0].min())/(dgm0[:,0].max() - dgm0[:,0].min())
                dgm0[:,1] = (dgm0[:,1] - dgm0[:,1].min())/(dgm0[:,1].max() - dgm0[:,1].min())
                for i in range(b.shape[2]//dgm0.shape[0]):
                    b[j,ii,i*dgm0.shape[0]:(i+1)*dgm0.shape[0],:] = dgm0
                b[j,ii,(i+1)*dgm0.shape[0]:,:] = dgm0[:b.shape[2] -(i+1)*dgm0.shape[0],: ]
        return b
    
    def buffer(self,data):
        '''
        

        Parameters
        ----------
        data : np.array.
            Input data

        Returns
        -------
        temp_buf : np.array
            Data with overlap.

        '''

        
        temp_buf = [data[i:i+self.duration] for i in range(0, len(data), (self.duration - int(np.ceil(self.data_overlap))))]
        
        temp_buf[self.number_of_segments-1] = np.pad(temp_buf[self.number_of_segments-1],
                                         (0, self.duration-temp_buf[self.number_of_segments-1].shape[0]),
                                     'constant')
        temp_buf = np.hstack(temp_buf[0:self.number_of_segments])
        return temp_buf 
    
    def data_segmentation(self,data, shift_len):
        '''
        

        Parameters
        ----------
        data : np.array.
            Input data with shape (n_channels, N_points, n_targets, n_trials).
        shift_len : int
            Window_len movement(second).

        Returns
        -------
        np.array
            Segmented data.

        '''
        
        num_sub = data.shape[0]
        num_classes = data.shape[3]
        num_chan = data.shape[1]
        num_trials = data.shape[4]

        self.duration = int(self.window_len*self.sample_rate)
        self.data_overlap = (self.window_len - shift_len)*self.sample_rate

        self.number_of_segments = int(math.ceil((data.shape[2] - self.data_overlap)/
                                           (self.duration - self.data_overlap)))

        DATA = dict()
        DATA_pers = dict()
        for sub in range(num_sub):
            subject_name_EEG = 'subject%d_EEG'%sub
            subject_name_label = 'subject%d_label'%sub
            if self.Purpose == 'Pretrain':
                d = data[sub,:,:,:,:]##### Format = n_channels, N_points, n_targets, n_trials
            elif self.Purpose == 'Target':
                d = data[sub,:,:self.sample_rate,:,:]##### Format = n_channels, N_points, n_targets, n_trials


            
            segmented_data = np.zeros((num_chan, self.duration * self.number_of_segments, num_classes, num_trials ))
            for target in range(num_classes):
                for trial in range(num_trials):
                    for channel in range(num_chan):
                    
                        segmented_data[channel,:, target, trial] = self.buffer(d[channel,: , target, trial])

            #segmented_data = np.reshape(segmented_data, (num_chan, self.duration, num_classes, num_trials * self.number_of_segments))
            #print(segmented_data.shape)
            d = self.data_conc(segmented_data)
            
            #d = filteration(d, lowcut, highcut, sampling_rate)
            final_data, labels = self.Data_split(d)
            labels = to_categorical(labels)
            #print(final_data.shape)
            if self.pers:
                persdata = self.convert_pers_to_matrix(final_data)
                DATA_pers[subject_name_EEG] = persdata

            if self.FFT:
                final_data = self.complex_spectrum_features(final_data)


            DATA[subject_name_EEG] = final_data
            DATA[subject_name_label] = labels


        if self.pers:
            return DATA, DATA_pers
        else:

            return DATA
        
    def Get_data(self,shift_len_train, shift_len_test):
        '''
        

        Parameters
        ----------
        shift_len_train : int
            Window_len movement for train data(second).
        shift_len_test : int
            Window_len movement for test data(second).

        Returns
        -------
        Dict
            Two dictionary containind train and test data with their labels.

        '''
        
        #shift_len_train = 1
        train = self.init_train
        
        #shift_len_test = 0.5
        test = self.init_test
        if self.pers:
            
            data_train, data_train_pers = self.data_segmentation(train , shift_len_train)
            data_test, data_test_pers = self.data_segmentation(test , shift_len_test)
            return data_train, data_train_pers, data_test, data_test_pers
        else:
            data_train = self.data_segmentation(train , shift_len_train)
            data_test = self.data_segmentation(test , shift_len_test)
            return data_train, data_test
        
        
    def concat_subs(self,shift_len_train, shift_len_test):
        '''
        

        Parameters
        ----------
        shift_len_train : int
            Window_len movement for train data(second).
        shift_len_test : int
            Window_len movement for test data(second).

        Returns
        -------
        np.array
            Train and test data afetr concatenating the subjects (2 subjects).

        '''

        
        if self.pers:
            data_train, data_train_pers, data_test, data_test_pers = self.Get_data(shift_len_train, shift_len_test)
            dt = np.concatenate((data_train['subject1_EEG'],data_train['subject2_EEG']), axis = 0)
            dt_pers = np.concatenate((data_train_pers['subject1_EEG'],data_train_pers['subject2_EEG']), axis = 0)
            label = np.concatenate((data_train['subject1_label'],data_train['subject2_label']), axis = 0)
            
            dte = np.concatenate((data_test['subject1_EEG'],data_test['subject2_EEG']), axis = 0)
            dte_pers = np.concatenate((data_test_pers['subject1_EEG'],data_test_pers['subject2_EEG']), axis = 0)
            label_te = np.concatenate((data_test['subject1_label'],data_test['subject2_label']), axis = 0)
            
            
            rng = np.random.default_rng()
            ind = np.arange(dt.shape[0])
            rng.shuffle(ind)
            dt = dt[ind]
            dt_pers = dt_pers[ind]
            label = label[ind]
            return dt, dt_pers, label, dte, dte_pers, label_te
        else:
            data_train, data_test = self.Get_data(shift_len_train, shift_len_test)
            dt = np.concatenate((data_train['subject1_EEG'],data_train['subject2_EEG']), axis = 0)
            label = np.concatenate((data_train['subject1_label'],data_train['subject2_label']), axis = 0)
            
            dte = np.concatenate((data_test['subject1_EEG'],data_test['subject2_EEG']), axis = 0)
            label_te = np.concatenate((data_test['subject1_label'],data_test['subject2_label']), axis = 0)
            
            
            rng = np.random.default_rng()
            ind = np.arange(dt.shape[0])
            rng.shuffle(ind)
            dt = dt[ind]
            label = label[ind]
            return dt, label, dte,label_te
            
            