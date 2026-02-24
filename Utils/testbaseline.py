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
from Models.GCTNet.models import DuoCL,GCTNet,basemodel
from Models.Generator import EEGDenoiseGenerator_NoSeq,EEGDenoiseGenerator,EEGDenoiseGeneratorv2
from Utils.test import test
from Utils.saveresult import write_to_excel
from Utils.dataprocess import data_preprocess,fix_random_seed
from Utils import Trainer
from etc.global_config import config


def run(basegenerator):
    torch.set_num_threads(5)

    devicesid= config['train_param']['cuda']
    if devicesid != 'cpu':
        devices = f"cuda:{devicesid}" if torch.cuda.is_available() else "cpu"
    else :
        devices ='cpu'
    print(devices)
    print(f"{'*' * 20} Current Algorithm usage: basline {'*' * 20}")

    
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
 


    start_time = time.time()
    
    if datasetid == 2 :
        Kf = 6
    elif datasetid == 1:
        Kf = 5
    else:
        Kf = 4
    for i in range (Kf):
        fix_random_seed(i)
        for Subject in range(1, Ns + 1):
            ACC_F,ACC_R=0,0
            print(f'Kf:{i}')
            print(f'subject:{Subject}')
            Data_Train, Data_Test,masks= EEGDataset.getSSVEPIntra(subject=Subject, train_ratio=ratio,kfold=i)[:]
            eeg_train_dataloader, eeg_test_dataloader, EEGData_Train, NOISYData_Train,EEGLabel_Train,EEGData_Test,NOISYData_Test, EEGLabel_Test= data_preprocess(Data_Train,Data_Test) 
            baseline = basegenerator #GCTNet ,CT_DCENet,EEGDNet
            if baseline == 'CT_DCENet':
                generator=CT_DCENet(
                    signal_len=Nt,
                    head_type_list=[0,0,1,1],
                    block_num=5
                )
            elif baseline =='EEGDNet':
                generator=DeT(
                    seq_len=Nt,      # = EEG 时间长度 T
                    patch_len=64,
                    depth=6,
                    heads=1
                )
            else:
                generator = pick_models(baseline,devices,Nt)
            pretrain = False
            if not pretrain:
                if baseline == 'CT_DCENet':
                    generator,_=Trainer.train_ct_dcenet_with_cls(
                        eeg_train_dataloader,
                        eeg_test_dataloader,
                        devices,
                        generator,
                        epochs_stage1=100,
                        epochs_stage2=100,
                        lambda_cls=1.0
                    )
                elif baseline == 'EEGDNet':
                    generator=Trainer.Train_EEGDNet(eeg_train_dataloader,
                    eeg_test_dataloader,
                    devices,
                    generator,
                    epochs=500,
                    lr=1e-3,
                    betas=(0.5, 0.9),
                    eps=1e-8,
                    lambda_cls=1.0
                    )
                else:
                    generator,model_d, test_acc_fake_cls, test_acc_real_cls = Trainer.train_GCTNet(
                        eeg_train_dataloader,
                        eeg_test_dataloader,
                        generator,
                        200,
                        devices)
                save_dir = "/data2/hzt/ssvep/baseline"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    generator,
                    os.path.join(
                        save_dir,
                        f"kf{i}set{datasetid}sub{Subject}-{noisetype}-{noise_ratio}-{snr_db}-{baseline}.pth"
                    )
                )
            else:
                generator = torch.load(os.path.join(
                        "/data2/hzt/ssvep/baseline",
                        f"kf{i}set{datasetid}sub{Subject}-{noisetype}-{noise_ratio}-{snr_db}-{baseline}.pth"
                    ), map_location=devices)
                generator = generator.to(devices)
            generator.eval()
            B, _, Nc, L = NOISYData_Train.shape
            x_train = NOISYData_Train.squeeze(1)
            x_train = x_train.reshape(-1, L).to(devices)
            with torch.no_grad():
                if baseline == 'CT_DCENet':
                    x_train_g = generator.denoise(x_train)
                elif baseline == 'GCTNet':
                    x_train_g = generator(x_train.unsqueeze(dim=1))
                else:
                    x_train_g = generator(x_train)
            x_train_g = x_train_g.reshape(B, Nc, L)
            x_train_g = x_train_g.unsqueeze(1)
            x_train_g = x_train_g.cpu().numpy()
            B2, _, Nc, L = NOISYData_Test.shape
            x_test = NOISYData_Test.squeeze(1)
            x_test = x_test.reshape(-1, L).to(devices)
            with torch.no_grad():
                if baseline == 'CT_DCENet':
                    x_test_g = generator.denoise(x_test)
                elif baseline == 'GCTNet':
                    x_test_g = generator(x_test.unsqueeze(dim=1))
                else:
                    x_test_g = generator(x_test)
            x_test_g = x_test_g.reshape(B2, Nc, L)
            x_test_g = x_test_g.unsqueeze(1)
            x_test_g = x_test_g.cpu().numpy()
            ACC = test(testmethod,config,devices, EEGData_Train, NOISYData_Train,x_train_g,EEGLabel_Train , EEGData_Test, NOISYData_Test,x_test_g,EEGLabel_Test)
            filename=f"result_baseline/set{datasetid}-{noisetype}-{baseline}.xlsx"
            sheetname=f"Sheet-{noise_ratio}-{snr_db}-{testmethod}"
            write_to_excel(i, Subject, ACC, (ACC_F,ACC_R),filename=filename, sheetname=sheetname) 
                
                

    end_time = time.time()

    print("cost_time:", end_time - start_time)
    
def pick_models(baseline,device, data_num=512):
    
    if baseline == 'SimpleCNN':
        model = basemodel.SimpleCNN(data_num).to(device)
                    
    elif baseline == 'FCNN':  
        model = basemodel.FCNN(data_num).to(device)
                
    elif baseline == 'ResCNN':
        model = basemodel.ResCNN(data_num).to(device)
    elif baseline == 'GCTNet':
        model = GCTNet.Generator(data_num).to(device)
    
    elif baseline == 'GeneratorCNN':
        model = GCTNet.GeneratorCNN(data_num).to(device)
        
    elif baseline == 'GeneratorTransformer':
        model = GCTNet.GeneratorTransformer(data_num).to(device)
    
    elif baseline == 'NovelCNN':
        model = basemodel.NovelCNN(data_num).to(device)
    
    elif baseline == 'DuoCL':
        model = DuoCL(data_num).to(device)
        
    else:
        print("model name is error!")
        pass
    return model



if __name__ == '__main__':
    os.chdir('/denonet')
    run('GCTNet')
