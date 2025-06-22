import argparse
import os

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pywt
import scipy.io as scio
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# function to apply different augmentations
def augment_ecg_signal(ecg_signal, sampling_rate=500):
    # Add Gaussian Noise
    noise_level = 0.01
    noise = np.random.normal(0, noise_level, len(ecg_signal))
    ecg_with_noise = ecg_signal + noise
    
    # Time Shifting
    shift = np.random.randint(-50, 50)  # Random shift between -50 and +50 samples
    ecg_time_shifted = np.roll(ecg_signal, shift)

    # Amplitude Scaling
    scale_factor = np.random.uniform(0.8, 1.2)  # Scale amplitude between 80% to 120%
    ecg_amplitude_scaled = ecg_signal * scale_factor

    # Time Warping
    warp_factor = 0.05
    indices = np.arange(len(ecg_signal))
    warp_indices = indices + warp_factor * np.sin(np.linspace(0, np.pi, len(ecg_signal)))
    ecg_time_warped = np.interp(indices, warp_indices, ecg_signal)

    # Frequency Shifting using NeuroKit2
    ecg_filtered = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='biosppy')
    
    return [ecg_with_noise,ecg_time_shifted,ecg_amplitude_scaled,ecg_time_warped,ecg_filtered]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
        
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--count")
    parser.add_argument("--aug_type")
    parser.add_argument("--to_aug")
    
    args = parser.parse_args()
    
    base_path = './'
    dataset_path =  args.input_folder # Training data
    
    classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
            'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
    ClassesNum = len(classes)

    X = list()
    y = list()

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
        
            data_train = scio.loadmat(os.path.join(root, name))
            
            # arr -> list
            data_arr = data_train.get('val')
            
            data_list = data_arr.tolist()
        
            X.append(data_list[0]) # [[……]] -> [ ]
            y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num('02' -> 1)

        
    X=np.array(X) # (1000, 3600)
    y=np.array(y) # (1000, )

    X = standardization(X)

    X = X.reshape((int(args.count), 3600))
    y = y.reshape((int(args.count)))
    scales = np.arange(1,200)

    # for i in range(1000):
    #     coef,freq = pywt.cwt(X[i],scales,'gaus1')
    #     plt.imshow(abs(coef),extent = [0,200,100,1],interpolation='bilinear',cmap='bone')
    #     plt.gca().invert_yaxis()
    #     plt.savefig(f'DatasetWave/{y[i]}/{i}.png')
    #     plt.close()

    # for i in range(1000):
        # plt.specgram(X[0],NFFT=256,Fs=600,noverlap=128,cmap='jet_r')
        # plt.colorbar()
        # plt.savefig(f'DatasetSpectro/{y[i]}/{i}.png')
        # plt.close()
        

    label=[]
    X = list(X)
    y = list(y)

    # for i in tqdm(range(1000), desc='Processing data', unit='file'):
    #     aug = augment_ecg_signal(X[i])
    #     X.extend(aug)
    #     y.extend(np.ones(5,dtype=int)*y[i])
        
    for i in tqdm(range(int(args.count) if int(args.to_aug) == 0 else 6*int(args.count)), desc='Saving data', unit='file'):
        coef, freq = pywt.cwt(X[i],scales,'gaus1')
        
        if args.aug_type == 'wavelet':
            # save wavelet as png:
            save_dir = f'../data/DatasetWaveNew_3/{y[i]}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.imshow(abs(coef), extent=[0, 200, 100, 1], interpolation='bilinear', cmap='bone')
            plt.gca().invert_yaxis()
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()
            
        elif args.aug_type == 'spectrogram':
            # save spectrogram as png:
            plt.specgram(X[0],NFFT=256,Fs=600,noverlap=128,cmap='jet_r')
            plt.colorbar()
            plt.savefig(f'../data/DatasetSpectro/{y[i]}/{i}.png')
            plt.close()
            
        elif args.aug_type == 'wavelet_mat':
            # save wavelet as matlab files:
            directory = f'../data/DatasetWave_1D_2/{y[i]}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            mat_filename = os.path.join(directory, f'{i}.mat')
            data_to_save = {
                'val': coef
                }
            scio.savemat(mat_filename, data_to_save)
    
if(__name__ == "__main__"):
    main()
