# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:14:42 2022

@author: Biodynamics
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert

#%% Method 1

FFT_PARAMS = {
    'resolution': 0.1,
    'start_frequency': 3.0,
    'end_frequency': 35.0,
    'sampling_rate': 500
}

def complex_spectrum_features(segmented_data, FFT_PARAMS):
    
    num_chan = segmented_data.shape[0]
    fft_len = segmented_data.shape[1]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros((segmented_data.shape[0], 2*(fft_index_end - fft_index_start), ))
    
    for channel_i in range(0, num_chan):
            temp_FFT = np.fft.fft(segmented_data[channel_i, :,], NFFT)/fft_len
            real_part = np.real(temp_FFT)
            imag_part = np.imag(temp_FFT)
            features_data[channel_i,:] = np.concatenate((
                real_part[fft_index_start:fft_index_end,], 
                imag_part[fft_index_start:fft_index_end,]), axis=0)
    
    return features_data


data = np.load('C:/Users/Biodynamics/Desktop/Data for my publication/Toyota Icon selection/Experiment 3/'+'RR1'+'.npy')
    
time_resol1 = np.linspace(3, 35,num=642)

#prepare test data from 15 trials selected from the last 

test_data = data[:,:,:,:,30:]
[N_block,N_channels,N_points,N_target,N_trials] = np.shape(test_data)

fft_data1 = complex_spectrum_features(test_data[0,:,:int(3*FFT_PARAMS['sampling_rate']),3,0],FFT_PARAMS)


analytic_signal = hilbert(fft_data1[6,:])
amplitude_envelope = np.abs(analytic_signal)

plt.plot(time_resol1,fft_data1[6,:])
plt.plot(time_resol1,amplitude_envelope)



#%% Method 2
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

high_idx, low_idx = hl_envelopes_idx(fft_data1[6,:],)


plt.plot(time_resol1,fft_data1[6,:])
plt.plot(time_resol1[low_idx],fft_data1[6,:][low_idx],linewidth=3)
plt.plot(time_resol1[high_idx],fft_data1[6,:][high_idx],linewidth=3)









