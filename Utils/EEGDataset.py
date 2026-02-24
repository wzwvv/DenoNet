import random
from torch.utils.data import Dataset
import torch
import scipy.io
import numpy as np
from scipy import signal
from scipy.io import loadmat
import os
from etc.global_config import config
import itertools
from Utils.dataprocess import add_noise

class getSSVEPIntra(Dataset):
   def __init__(self, subject=1, train_ratio=0.8,kfold = 0):
       super(getSSVEPIntra, self).__init__()
       self.datasetid = config["train_param"]["datasets"]
       self.Nh = config[f"data_param{self.datasetid}"]["Nh"]  # number of trials
       self.Nc = config[f"data_param{self.datasetid}"]["Nc"] # number of channels
       self.Nt = config[f"data_param{self.datasetid}"]["Nt"]  # number of time points
       self.Nf = config[f"data_param{self.datasetid}"]["Nf"]  # number of target frequency
       self.Fs = config[f"data_param{self.datasetid}"]["Fs"] # Sample Frequency
       self.ws = config["train_param"]["ws"]   # window size of ssvep
       self.Nm = config["model_param"]["Nm"]
       noisetype=config['train_param']['noisetype']
       noise_ratio=config['train_param']['noise_ratio']
       snr_db=config['train_param']['snr_db']
       self.T = int(self.Fs * self.ws)
       self.subject = subject  # current subject
       if self.datasetid==1:
           self.data = self.load_1Data(data_path='/data2/hzt/ssvep/benchmark12',segment=self.ws)
           self.indexid =self.intra_benchmark12_split()
       elif self.datasetid == 2:
           self.data = self.load_2Data('/data2/hzt/ssvep/benchmark40',segment=self.ws)
           self.indexid =self.intra_benchmark40_split(train_ratio)
       elif self.datasetid == 3:
           self.data = self.load_3Data('/data2/hzt/ssvep/BETA',segment=self.ws)
           self.indexid =self.intra_benchmark40_split(train_ratio)
       self.eeg_data=self.data[0]
       self.noisy_data,self.masks=add_noise(self.eeg_data,noisetype=noisetype,noise_ratio=noise_ratio,snr_db=snr_db)
       self.label_data=self.data[1]
       self.eeg_data=self.filter_bank(self.eeg_data)#(self.Nh, max(self.Nm,1), self.Nc, self.T))
       self.noisy_data=self.filter_bank(self.noisy_data)
       self.train_idx = []
       self.test_idx = []
       for i in range(0, self.Nh, self.Nh // self.Nf):
           for j in range(self.Nh // self.Nf):
               if train_ratio < 0.5 :
                if j in self.indexid[kfold]:
                    self.train_idx.append(i + j)
                else:
                    self.test_idx.append(i + j)
               else:
                   if j in self.indexid[kfold]:
                    self.test_idx.append(i + j)
                   else:
                    self.train_idx.append(i + j)
       self.eeg_data_train = self.eeg_data[self.train_idx]
       self.noisy_eeg_train = self.noisy_data[self.train_idx]
       self.label_data_train = self.label_data[self.train_idx]
       self.eeg_data_test = self.eeg_data[self.test_idx]
       self.noisy_eeg_test = self.noisy_data[self.test_idx]
       self.label_data_test = self.label_data[self.test_idx]

   def __getitem__(self,index):
       return (self.eeg_data_train, self.noisy_eeg_train,self.label_data_train),(self.eeg_data_test,self.noisy_eeg_test,self.label_data_test),self.masks

   def __len__(self):
       return len(self.label_data)

   def filter_bank(self,eeg):
       result = np.zeros((self.Nh, max(self.Nm,1), self.Nc, self.T))

       nyq = self.Fs / 2
       
       if self.datasetid == 1:
           passband = [9, 18, 27, 36, 45, 54]
           stopband = [7, 15, 19, 28, 37, 46]

       else:
           passband = [8, 16, 24, 32, 40, 48]
           stopband = [6, 12, 18, 26, 34, 42]
    #    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    #    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]

       highcut_pass, highcut_stop = 80,90

       gpass, gstop, Rp = 3, 40, 0.5
       if self.Nm:
           for i in range(self.Nm):
               Wp = [passband[i] / nyq, highcut_pass / nyq]
               Ws = [stopband[i] / nyq, highcut_stop / nyq]
               [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
               [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')

               data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
               result[:, i, :, :] = data
       else:
           result[:, 0, :, :]=eeg
       return result

   def load_1Data(self, data_path='Dataset/benchmark12',decay=0.14, segment=1):
        subjectfile = loadmat(os.path.join(data_path, f'S{self.subject}.mat'))
        samples = subjectfile['eeg']  # (12, 8, 1024, 15)
        eeg_data = samples[0, :, :, :]  # (8, 1024, 15)
        label_data = np.zeros((180, 1))
        for i in range(1, 12):
            eeg_data = np.concatenate([eeg_data, samples[i, :, :, :]], axis=2)
            label_data[i * 15:(i + 1) * 15] = i
        eeg_data = eeg_data.transpose([2, 0, 1])  # (8, 1114, 180) -> (180, 8, 1024)
        eeg_data = eeg_data[:, :, int((decay) * 256):int(( decay + segment) * 256)]
        return eeg_data,label_data

   def load_2Data(self, data_path='Dataset/benchmark40', decay=0.14, segment=1):
       data = loadmat(os.path.join(data_path, f'S{self.subject}.mat'))
       x0 = data['data'][[47, 53, 54, 55, 56, 57, 60, 61, 62], int((0.5 + decay) * 250):int((0.5 + decay + segment) * 250),:,:]#(64, 250, 40, 6)
       x0 = x0.transpose(2,0,1,3) #(40, 64, 250, 6)
       eeg_data = x0[0] #(64, 250, 6)
       label_data = np.zeros((720, 1))# )
       for i in range(1, self.Nf):
            eeg_data = np.concatenate([eeg_data, x0[i]],axis=2)
            label_data[i * 6:(i + 1) * 6] = i
       eeg_data = eeg_data.transpose([2, 0, 1])
       return eeg_data[:240], label_data[:240]
   def intra_benchmark40_split(self,ratio=0.85):
    num_blocks = 6
    num_groups = 1
    blocks_per_group = num_blocks // num_groups  # 6

    num_samples = 6 - int(6 * ratio) 
    groups = [list(range(i * blocks_per_group, (i + 1) * blocks_per_group)) for i in range(num_groups)]

    combinations_per_group = [list(itertools.combinations(g, num_samples)) for g in groups]
    min_len = min(len(c) for c in combinations_per_group)
    merged = [
        combinations_per_group[0][i]
        for i in range(min_len)
    ]
    return merged
   def intra_benchmark12_split(self,num_splits=5):
        splits=[]
        for i in range(num_splits):
            train_subjects = [ i*3+j for j in range(3)]

            splits.append(train_subjects)
        return splits