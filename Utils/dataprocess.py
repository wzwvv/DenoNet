from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from etc.global_config import config
from pandas.io.sas.sas_constants import dataset_length
from types import SimpleNamespace
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os

def fix_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 如果有多卡

    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # cuDNN 相关（可复现性最重要）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def time_noise(x, noise_ratio=0.1, snr_db=10):
    """
    时间加噪: 在某个时间段的所有通道加噪声 (同一时间区间)
    Args:
        x: [B, 1, C, T]
        time_noise_ratio: 时间段比例
        noise_std: 高斯噪声标准差
    Returns:
        x: 加噪后的数据
        masks: 噪声掩码 (bool) [B, 1, C, T]
    """
    B, C, T = x.shape
    time_len = int(T * noise_ratio)
    masks = torch.zeros_like(x, dtype=torch.bool)
    for b in range(B):
        start = random.randint(0, T - time_len)
        end = start + time_len
        # 所有通道共享同一个噪声段
        for c in range(C):
            # 计算信号功率
            Ps = torch.mean(x[b,c, :] ** 2)
            # 计算噪声功率
            Pn = Ps / (10 ** (snr_db / 10))
            # 生成高斯噪声
            noise_std = torch.sqrt(Pn)
            noise = torch.randn(end - start, device=x.device) * noise_std
            x[b, c, start:end] += noise
        masks[b, :, start:end] = True

    return x, masks


def space_noise(x, noise_ratio=0.2, snr_db=10):
    B, C, T = x.shape
    masks = torch.zeros_like(x, dtype=torch.bool)
    num_channels_noisy = int(C * noise_ratio)
    for b in range(B):
        noisy_channels = random.sample(range(C), num_channels_noisy)
        for c in noisy_channels:
            Ps = torch.mean(x[b, c, :] ** 2)
            # 计算噪声功率
            Pn = Ps / (10 ** (snr_db / 10))
            # 生成高斯噪声
            noise_std = torch.sqrt(Pn)
            noise = torch.randn(T, device=x.device) * noise_std
            x[b, c, :] += noise
            masks[b, c, :] = True
    return x, masks

def freq_noise(x, noise_ratio=0.2, snr_db=10, freq_band=None):
    """
    频率加噪: 在频域中添加高斯噪声 (可指定频段)
    Args:
        x: [B, 1, C, T]  EEG 数据
        noise_ratio: 噪声频率占比 (0~1)，如果未指定 freq_band 时使用
        noise_std: 噪声标准差
        freq_band: tuple(低频Hz, 高频Hz)，可选
    Returns:
        x_noisy: 加噪后的信号
        masks: 噪声掩码 (bool) [B, 1, C, T]
    """
    B,  C, T = x.shape
    masks = torch.zeros_like(x, dtype=torch.bool)

    # 频率索引
    freq_indices = torch.fft.fftfreq(T)  # 范围 -0.5~0.5，对应归一化频率

    for b in range(B):
        band_len = int(T * noise_ratio / 4)
        start = random.randint(0, T // 4 - band_len)
        for c in range(C):

            Xf = torch.fft.fft(x[b, c, :])
            # 对每个通道做 FFT
            Ps = torch.mean(torch.abs(Xf[:T // 4]) ** 2)
            # 计算噪声功率
            Pn = Ps / (10 ** (snr_db / 10))
            # 生成高斯噪声
            noise_std = torch.sqrt(Pn)

            # 噪声区间
            if freq_band is not None:
                f_low, f_high = freq_band
                # 假设采样率未知，这里 freq_indices 是归一化频率 [-0.5, 0.5)
                mask = (torch.abs(freq_indices) >= f_low) & (torch.abs(freq_indices) <= f_high)
            else:
                # 随机选取一段频率区间
                mask = torch.zeros(T, dtype=torch.bool, device=x.device)
                mask[start:start+band_len] = True
                mask[-(start+band_len):-start] = True  # 对称部分

            # 生成频域噪声（复数）
            noise_real = torch.randn(T, device=x.device) * noise_std
            noise_imag = torch.randn(T, device=x.device) * noise_std
            noise_freq = noise_real + 1j * noise_imag

            # 应用噪声
            Xf_noisy = Xf + noise_freq * mask

            # IFFT 返回时域信号
            x_noisy_t = torch.fft.ifft(Xf_noisy).real
            x[b, c, :] = x_noisy_t
            masks[b,  c, :] = mask
    return x, masks

def add_noise(x, noisetype='time', noise_ratio=0.25, snr_db=5):
    eeg=torch.tensor(x.copy())
    masks = torch.zeros_like(eeg, dtype=torch.bool)
    if noisetype=='time':
        eeg, masks = time_noise(eeg, noise_ratio, snr_db)
    if noisetype == 'space':
        eeg, masks = space_noise(eeg, noise_ratio, snr_db)
    if noisetype == 'freq':
        eeg, masks = freq_noise(eeg, noise_ratio, snr_db)
    elif noisetype == 'random':
        for ii in range(len(x)):
            choice = random.choice(['time', 'space', 'freq'])
            if choice == 'time':
                eeg[ii:ii+1], masks[ii] = time_noise(eeg[ii:ii+1], noise_ratio, snr_db)
                
            elif choice == 'space':
                eeg[ii:ii+1], masks[ii] = space_noise(eeg[ii:ii+1], noise_ratio, snr_db)
            else:
                eeg[ii:ii+1], masks[ii] = freq_noise(eeg[ii:ii+1], noise_ratio, snr_db)
    elif noisetype == 'all':
        eeg, mask1 = time_noise(eeg, noise_ratio, snr_db)
        eeg, mask2= space_noise(eeg, noise_ratio, snr_db)
        eeg, mask3 = freq_noise(eeg, noise_ratio, snr_db)
        masks = mask1 | mask2 | mask3
    #visualize_eeg_multichannel(raw_eeg_train, x[:,:,:,:], masks, sample_idx=0)
    # if len(idx_space) > 0:
    #     visualize_eeg_multichannel(raw_eeg_train, X_train_noise[:,:,:Nc,:], masks, sample_idx=idx_space[0].item())
    # if len(idx_both) > 0:
    #     visualize_eeg_multichannel(raw_eeg_train, X_train_noise[:,:,:Nc,:], masks, sample_idx=idx_both[0].item())
    # if len(idx_none) > 0:
    #     visualize_eeg_multichannel(raw_eeg_train, X_train_noise[:, :, :Nc, :], masks, sample_idx=idx_none[0].item())
    return eeg,masks

def data_preprocess(EEGData_Train, EEGData_Test):

    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    datasetid = config["train_param"]['datasets']
    bz = config["train_param"]['bz']
    Nm = config["model_param"]["Nm"]
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    noisetype = config["train_param"]['noisetype']
    noise_ratio = config["train_param"]['noise_ratio']
    snr_db = config["train_param"]['snr_db']

    '''Loading Training Data'''
    EEGData_Train, NOISYData_Train,EEGLabel_Train = EEGData_Train[:]
    EEGData_Train = torch.tensor(EEGData_Train,dtype=torch.float)
    NOISYData_Train = torch.tensor(NOISYData_Train,dtype=torch.float)
    EEGLabel_Train = torch.tensor(EEGLabel_Train)


    #EEGData_Train=EEGData_Train[:,:,[47, 53, 54, 55, 56, 57, 60, 61, 62],:]
    EEGData_Train = EEGData_Train.flatten(start_dim=1, end_dim=2)
    EEGData_Train = EEGData_Train.unsqueeze(1)
    NOISYData_Train = NOISYData_Train.flatten(start_dim=1, end_dim=2)
    NOISYData_Train = NOISYData_Train.unsqueeze(1)
    #X_train_noise, masks= add_noise(raw_eeg_train,noisetype=noisetype,noise_ratio=noise_ratio,snr_db=snr_db)



    print("EEGData_Train.shape", EEGData_Train.shape)
    # print("EEGnoise_Train.shape", X_train_noise.shape)
    # print("EEGLabel_Train.shape", EEGLabel_Train.shape)
    EEGData_Train_set = torch.utils.data.TensorDataset(EEGData_Train,NOISYData_Train,EEGLabel_Train)

    '''Loading Testing Data'''
    EEGData_Test,NOISYData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = torch.tensor(EEGData_Test,dtype=torch.float)
    NOISYData_Test = torch.tensor(NOISYData_Test,dtype=torch.float)
    EEGLabel_Test = torch.tensor(EEGLabel_Test)
    
    
    
    #EEGData_Test=EEGData_Test[:,:,[47, 53, 54, 55, 56, 57, 60, 61, 62],:]
    EEGData_Test = EEGData_Test.flatten(start_dim=1, end_dim=2)
    EEGData_Test = EEGData_Test.unsqueeze(1)
    NOISYData_Test = NOISYData_Test.flatten(start_dim=1, end_dim=2)
    NOISYData_Test = NOISYData_Test.unsqueeze(1)
    #X_test_noise, masks= add_noise(raw_eeg_test,noisetype=noisetype,noise_ratio=noise_ratio,snr_db=snr_db)
    # print("EEGData_Test.shape", EEGData_Test.shape)
    # print("EEGnoise_Test.shape", X_test_noise.shape)
    # print("EEGLabel_Test.shape", EEGLabel_Test.shape)
    EEGData_Test_set = torch.utils.data.TensorDataset(EEGData_Test, NOISYData_Test,EEGLabel_Test)

    eeg_train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train_set, batch_size=bz, shuffle=True,
                                                   drop_last=True)
    eeg_test_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test_set, batch_size=bz, shuffle=False,
                                                   drop_last=True)
    return eeg_train_dataloader, eeg_test_dataloader,EEGData_Train, NOISYData_Train,EEGLabel_Train,EEGData_Test,NOISYData_Test, EEGLabel_Test
def visualize_eeg_multichannel(x_clean, x_noisy, mask, sample_idx=0, channel_names=None, offset=10.0):
    """
    将某个样本的多个通道脑电数据绘制在同一张图里 (类似EEG显示)

    Args:
        x_clean: 原始信号 [B, 1, C, T]
        x_noisy: 加噪信号 [B, 1, C, T]
        mask: 加噪掩码 [B, 1, C, T]
        sample_idx: 样本索引
        channel_names: 通道名字 (list)，默认为 Ch0, Ch1...
        offset: 通道之间的垂直偏移量
    """
    clean = x_clean[sample_idx, 0].cpu().numpy()  # [C, T]
    noisy = x_noisy[sample_idx, 0].cpu().numpy()  # [C, T]
    mask_c = mask[sample_idx, 0].cpu().numpy()  # [C, T]

    C, T = clean.shape
    t = np.arange(T)

    if channel_names is None:
        channel_names = [f"Ch{c}" for c in range(C)]

    plt.figure(figsize=(15, 10))

    for c in range(C):
        baseline = c * offset
        # 原始信号：整条黑线
        plt.plot(t, clean[c] + baseline, color="black", linewidth=1)
        # 加噪部分：红线 (画 noisy)
        plt.fill_between(t, clean[c] + baseline, noisy[c] + baseline,
                         where=mask_c[c], color="red", alpha=0.3)

        # 标注通道名
        plt.text(-T * 0.02, baseline, channel_names[c], va='center', fontsize=8)

    plt.title(f"EEG Sample {sample_idx} (Black=Original, Red=Noisy)")
    plt.xlabel("Time")
    plt.ylabel("Channels")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    # ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5','POz']
    # cmap = get_cmap('viridis', 13)
    # x_clean=x_clean[0,0,:,:]
    # n_points = x_clean.shape[1]
    # start_idx = int(0.25 * n_points)
    # end_idx = int(0.50 * n_points)
    # t = np.arange(x_clean.shape[1]) / 700  # 创建时间轴
    # for i in range(x_clean.shape[0]):
    #     ax = plt.subplot(x_clean.shape[0], 1, i + 1)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     if i ==2 :
    #         ax.plot(t, x_clean[2], color='red', linewidth=2)  # cmap(i)
    #     else:
    #         ax.plot(t, x_clean[i], color=cmap(i), linewidth=2)  # cmap(i)
    #     ax.plot(t[start_idx:end_idx], x_clean[i][start_idx:end_idx], color='red', linewidth=2)

    #     # ax.plot(t, sample_dataAug[i], label=f'DWT Aug', color='red')
    #     ax.set_xlim([0, x_clean.shape[1] / 256])  # 根据数据长度和采样频率设置x轴范围
    #     ax.set_ylabel(ch_names[i], fontsize=18, rotation=0, labelpad=20, verticalalignment='center')  # 纵坐标
    #     ax.set_yticklabels([])  # 对非最后一个子图隐藏x轴标签
    #     ax.yaxis.set_ticks([])
    #     # if i == sample_data.shape[0] - 1:
    #     #     ax.set_xlabel('Time (seconds)', fontsize=20)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.xaxis.set_ticks([])
    #     ax.set_xticklabels([])  # 对非最后一个子图隐藏x轴标签
    # plt.subplots_adjust(hspace=0.)  # hspace 控制子图间的垂直间距
    # plt.tight_layout()
    # plt.show()
    plt.savefig("your_filename.png", dpi=300, bbox_inches="tight")