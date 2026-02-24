import sys
import time
import torch
from setuptools.wheel import unpack
import numpy as  np
import os
import Utils.EEGDataset as EEGDataset
from Models import Generator
from Models.CT_DCENet.models import CT_DCENet
from Models.EEGDNet.Network_structure import DeT
from Utils.test import test
from Utils.saveresult import write_to_excel
from Models.Generator import EEGDenoiseGenerator_NoSeq,EEGDenoiseGenerator,EEGDenoiseGeneratorv2
from Utils.dataprocess import data_preprocess,fix_random_seed
from Utils import Trainer
from etc.global_config import config
import Utils.testbaseline.run as runbaseline
def run(basegenerator=None):
    torch.set_num_threads(5)
    devicesid= config['train_param']['cuda']
    if devicesid != 'cpu':
        devices = f"cuda:{devicesid}" if torch.cuda.is_available() else "cpu"
    else :
        devices ='cpu'
    print(devices)
    print(f"{'*' * 20} Current Algorithm usage: GAN {'*' * 20}")

    
    datasetid = config["train_param"]['datasets']
    ratio=config['train_param']['ratio']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    ws = config['train_param']['ws']
    Nt = int(ws*Fs)
    
    pretrain=config['train_param']['pre_train']
    testmethod = config['train_param']['testmethod']
    
    noisetype=config['train_param']['noisetype']
    noise_ratio=config['train_param']['noise_ratio']
    snr_db=config['train_param']['snr_db']
    
    lambda_G = config['gan_model_param']['lambda_G']
    lambda_vae = config['gan_model_param']['lambda_vae']
    lambda_kl = config['gan_model_param']['lambda_kl']


    start_time = time.time()
    
    if datasetid == 2 :
        Kf = 6
    elif datasetid == 1:
        Kf = 5
    else:
        Kf = 4
    for i in range (1,Kf):
        fix_random_seed(i)
        for Subject in range(1, Ns + 1):
            ACC_F,ACC_R=0,0
            print(f'Kf:{i}')
            print(f'subject:{Subject}')
            Data_Train, Data_Test,masks= EEGDataset.getSSVEPIntra(subject=Subject, train_ratio=ratio,kfold=i)[:]
            eeg_train_dataloader, eeg_test_dataloader, EEGData_Train, NOISYData_Train,EEGLabel_Train,EEGData_Test,NOISYData_Test, EEGLabel_Test= data_preprocess(Data_Train,Data_Test) 
            if pretrain:
                generator = Generator.EEGDenoiseGeneratorv2(Nc*Nm,Nt).to(devices)
                generator.load_state_dict(torch.load(f"/data2/hzt/ssvep/denonetv2/kf{i}set{datasetid}sub{Subject}-{noisetype}-{noise_ratio}-{snr_db}-generator-{lambda_G}-{lambda_vae}-{lambda_kl}.pth"))
            else:
                if basegenerator == 'CNN': #CNN,CNN_Trans,CNN_LSTM
                    ACC_F,ACC_R,generator,discriminator= Trainer.train_gan(eeg_train_dataloader, eeg_test_dataloader, devices,basegenerator=EEGDenoiseGenerator_NoSeq(Nc*Nm))
                elif basegenerator == 'CNN_Trans': 
                    ACC_F,ACC_R,generator,discriminator= Trainer.train_gan(eeg_train_dataloader, eeg_test_dataloader, devices,basegenerator=EEGDenoiseGeneratorv2(Nc*Nm,Nt))
                elif basegenerator == 'CNN_LSTM': 
                    ACC_F,ACC_R,generator,discriminator= Trainer.train_gan(eeg_train_dataloader, eeg_test_dataloader, devices,basegenerator=EEGDenoiseGenerator(Nc*Nm,Nt))
                else:
                    ACC_F,ACC_R,generator,discriminator= Trainer.train_gan(eeg_train_dataloader, eeg_test_dataloader, devices)
                    
            
            _,_,_,_,x_train_g = generator(NOISYData_Train.to(devices))
            x_train_g=x_train_g.cpu().detach().numpy()
            _,_,_,_,x_test_g = generator(NOISYData_Test.to(devices))
            x_test_g=x_test_g.cpu().detach().numpy()
              
            ACC = test(testmethod,config,devices, EEGData_Train, NOISYData_Train,x_train_g,EEGLabel_Train , EEGData_Test, NOISYData_Test,x_test_g,EEGLabel_Test)
            filename=f"results/set{datasetid}-{noisetype}-{config['train_param']['savefilename']}"
            sheetname=f"Sheet-{noise_ratio}-{snr_db}-{testmethod}-{lambda_G}-{lambda_vae}-{lambda_kl}"
            write_to_excel(i, Subject, ACC, (ACC_F,ACC_R),filename=filename, sheetname=sheetname)
                

    end_time = time.time()

    print("cost_time:", end_time - start_time)


if __name__ == '__main__':
    os.chdir('/denonet')
    baseline= False
    basegenerator = None # CNN,CNN_Trans,CNN_LSTM,GCTNet ,CT_DCENet,EEGDNet
    if not baseline :   
        run()
    else:
        if basegenerator in ['CNN', 'CNN_Trans', 'CNN_LSTM']:
            run(basegenerator)
        else:
            runbaseline(basegenerator)
            
