import numpy as np
from types import SimpleNamespace

from Models.KNoW.FBCCA import FBCCA
from Models.KNoW.TRCA import TRCA
from Models.DeepL import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN
from Utils import Constraint, LossFunction, Script
import torch
import torch.nn as nn
import time
def test(testmethod,config,devices, train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label):
    if testmethod == "FBCCA":
        ACC = fbcca_evaluate(config,train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label)
    elif testmethod == 'TRCA':
        ACC = trca_evaluate(config,train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label)
    else:
        ACC = model_evaluate(config,devices,train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label)
    return ACC
def prepare_inputs(arr):
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    return np.array(arr,dtype=np.float64)      

def fbcca_evaluate(config, train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label):
    """
    FBCCA 去噪前后准确率测试函数（兼容 tensor / numpy / TensorDataset）
    输入数据必须为 shape [N,1,C,T] 或 [N,C,T]
    """

    train_X_clean = prepare_inputs(train_data)
    train_X_noisy = prepare_inputs(train_noise)
    train_X_g = prepare_inputs(train_g)
    train_y = np.array(train_label) if train_label is not None else np.array(train_label_ds).reshape(-1)
    train_y =  np.array(train_y).flatten()

    test_X_clean = prepare_inputs(test_data)
    test_X_noisy = prepare_inputs(test_noise)
    test_X_g = prepare_inputs(test_g)
    test_y = np.array(test_label) if test_label is not None else np.array(test_label_ds).reshape(-1)
    test_y= np.array(test_y).flatten()

    # ---------------------------
    # 读取配置参数
    # ---------------------------
    datasetid = config["train_param"]['datasets']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    Nf = config[f"data_param{datasetid}"]['Nf']
    Nh = config[f"data_param{datasetid}"]['Nh']
    ws = config["train_param"]['ws']
    initfreq = config[f"data_param{datasetid}"]['initfreq']
    deltafreq = config[f"data_param{datasetid}"]['deltafreq']

    opt = SimpleNamespace(Fs=Fs, ws=ws, Nm=Nm, Nc=Nc, Nf=Nf, dataset='your_dataset',
                          Nh=Nh, is_ensemble=False)
    targets = initfreq + np.arange(Nf) * deltafreq
    fbcca = FBCCA(opt)
    def dummy_filter_bank(eeg):
        segments, total_channels, T = eeg.shape
        assert total_channels == Nc * Nm, f"通道数量不匹配: {total_channels} != {Nc}*{Nm}"
        eeg = eeg.reshape(segments, Nm, Nc, T)
        return eeg
    fbcca.filter_bank = dummy_filter_bank
    pred_clean = fbcca.fbcca_classify(targets, test_X_clean, num_harmonics=Nm,train_labels=train_y,
                                                   train_data=train_X_clean, template=True)

    pred_clean = np.array(pred_clean).flatten()
    acc_clean = np.mean(test_y== pred_clean)
    print(f"FBCCA accuracy on CLEAN test data: {acc_clean:.4f} ({np.sum(test_y==pred_clean)}/{len(test_y)})")


    pred_noise = fbcca.fbcca_classify(targets, test_X_noisy, num_harmonics=Nm,train_labels=train_y,
                                                    train_data=train_X_noisy, template=True)
    pred_noise = np.array(pred_noise).flatten()
    acc_noise = np.mean(test_y== pred_noise)
    print(f"FBCCA accuracy on NOISY test data: {acc_noise:.4f} ({np.sum(test_y==pred_noise)}/{len(test_y)})")

    pred_g = fbcca.fbcca_classify(targets, test_X_g, num_harmonics=Nm, train_labels=train_y,
                                      train_data=train_X_g, template=True)
    pred_g = np.array(pred_g).flatten()
    acc_g = np.mean(test_y == pred_g)
    print(f"FBCCA accuracy on generator test data: {acc_g:.4f} ({np.sum(test_y == pred_g)}/{len(test_y)})")

    pred_cn = fbcca.fbcca_classify(targets,  np.concatenate([test_X_clean,test_X_noisy],axis=-1), num_harmonics=Nm,train_labels=train_y,
                                                   train_data=np.concatenate([train_X_clean,train_X_noisy],axis=-1), template=True)
    pred_cn = np.array(pred_cn).flatten()
    acc_cn = np.mean(test_y== pred_cn)
    print(f"FBCCA accuracy on C+N test data: {acc_cn:.4f} ({np.sum(test_y==pred_cn)}/{len(test_y)})")

    pred_cg = fbcca.fbcca_classify(targets, np.concatenate([test_X_clean,test_X_g],axis=-1),num_harmonics=Nm, train_labels=train_y,
                                   train_data=np.concatenate([train_X_clean, train_X_g], axis=-1), template=True)

    pred_cg = np.array(pred_cg).flatten()
    acc_cg = np.mean(test_y == pred_cg)
    print(f"FBCCA accuracy on C+G test data: {acc_cg:.4f} ({np.sum(test_y == pred_cg)}/{len(test_y)})")

    pred_ng = fbcca.fbcca_classify(targets, np.concatenate([test_X_noisy, test_X_g], axis=-1), num_harmonics=Nm,
                                   train_labels=train_y,
                                   train_data=np.concatenate([train_X_noisy, train_X_g], axis=-1), template=True)

    pred_ng = np.array(pred_ng).flatten()
    acc_ng = np.mean(test_y == pred_ng)
    print(f"FBCCA accuracy on N+G test data: {acc_ng:.4f} ({np.sum(test_y == pred_ng)}/{len(test_y)})")


    return acc_clean,acc_noise,acc_g,acc_ng,acc_cn, acc_cg

def trca_evaluate(config, train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label):
    """
    FBCCA 去噪前后准确率测试函数（兼容 tensor / numpy / TensorDataset）
    输入数据必须为 shape [N,1,C,T] 或 [N,C,T]
    """
    train_X_clean = prepare_inputs(train_data)
    train_X_noisy = prepare_inputs(train_noise)
    train_X_g = prepare_inputs(train_g)
    train_y = np.array(train_label) if train_label is not None else np.array(train_label_ds).reshape(-1)
    train_y =  np.array(train_y).flatten()

    test_X_clean = prepare_inputs(test_data)
    test_X_noisy = prepare_inputs(test_noise)
    test_X_g = prepare_inputs(test_g)

    test_y = np.array(test_label) if test_label is not None else np.array(test_label_ds).reshape(-1)
    test_y= np.array(test_y).flatten()

    train_X_cg=np.concatenate([train_X_clean, train_X_g], axis=-1)
    test_X_cg=np.concatenate([test_X_clean, test_X_g], axis=-1)

    train_X_cn=np.concatenate([train_X_clean, train_X_noisy], axis=-1)
    test_X_cn=np.concatenate([test_X_clean, test_X_noisy], axis=-1)

    train_X_ng=np.concatenate([train_X_noisy, train_X_g], axis=-1)
    test_X_ng=np.concatenate([test_X_noisy, test_X_g], axis=-1)
    # ---------------------------
    # 读取配置参数
    # ---------------------------
    datasetid = config["train_param"]['datasets']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    Nf = config[f"data_param{datasetid}"]['Nf']
    Nh = config[f"data_param{datasetid}"]['Nh']
    ws = config["train_param"]['ws']
    Nt = int(ws*Fs)

    test_y = test_y.reshape(Nf, -1)

    train_X_clean=train_X_clean.reshape(-1, Nm, Nc, Nt)
    train_X_clean_new = np.transpose(train_X_clean.reshape(Nf, -1, Nm, Nc, Nt),
                                       (0, 2, 3, 4, 1))
    test_X_clean = np.transpose(test_X_clean.reshape(Nf, -1, Nm, Nc, Nt),
                                      (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_clean_new)
    knowlayer.trca(train_X_clean, train_y)
    acc_clean =knowlayer.test_trca(test_X_clean, test_y)
    print(f"TRCA accuracy on CLEAN test data: {acc_clean:.4f} ")

    train_X_noisy = train_X_noisy.reshape(-1, Nm, Nc, Nt)
    train_X_noisy_new = np.transpose(train_X_noisy.reshape(Nf, -1, Nm, Nc, Nt),
                                     (0, 2, 3, 4, 1))
    test_X_noisy = np.transpose(test_X_noisy.reshape(Nf, -1, Nm, Nc, Nt),
                                (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_noisy_new)
    knowlayer.trca(train_X_noisy, train_y)
    acc_noisy = knowlayer.test_trca(test_X_noisy, test_y)
    print(f"TRCA accuracy on NOISY test data: {acc_noisy:.4f} ")

    train_X_g = train_X_g.reshape(-1, Nm, Nc, Nt)
    train_X_g_new = np.transpose(train_X_g.reshape(Nf, -1, Nm, Nc, Nt),
                                     (0, 2, 3, 4, 1))
    test_X_g = np.transpose(test_X_g.reshape(Nf, -1, Nm, Nc, Nt),
                                (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_g_new)
    knowlayer.trca(train_X_g, train_y)
    acc_g = knowlayer.test_trca(test_X_g, test_y)
    print(f"TRCA accuracy on GENERATOR test data: {acc_g:.4f} ")


    train_X_ng = train_X_ng.reshape(-1, Nm, Nc, 2*Nt)
    train_X_ng_new = np.transpose(train_X_ng.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                     (0, 2, 3, 4, 1))
    test_X_ng = np.transpose(test_X_ng.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_ng_new)
    knowlayer.trca(train_X_ng, train_y)
    acc_ng = knowlayer.test_trca(test_X_ng, test_y)
    print(f"TRCA accuracy on n+g test data: {acc_ng:.4f} ")

    train_X_cn = train_X_cn.reshape(-1, Nm, Nc, 2*Nt)
    train_X_cn_new = np.transpose(train_X_cn.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                     (0, 2, 3, 4, 1))
    test_X_cn = np.transpose(test_X_cn.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_cn_new)
    knowlayer.trca(train_X_cn, train_y)
    acc_cn = knowlayer.test_trca(test_X_cn, test_y)
    print(f"TRCA accuracy on c+n test data: {acc_cn:.4f} ")

    train_X_cg = train_X_cg.reshape(-1, Nm, Nc, 2*Nt)
    train_X_cg_new = np.transpose(train_X_cg.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                     (0, 2, 3, 4, 1))
    test_X_cg = np.transpose(test_X_cg.reshape(Nf, -1, Nm, Nc, 2*Nt),
                                (0, 2, 3, 4, 1))
    knowlayer = TRCA(train_X_cg_new)
    knowlayer.trca(train_X_cg, train_y)
    acc_cg = knowlayer.test_trca(test_X_cg, test_y)
    print(f"TRCA accuracy on c+g test data: {acc_cg:.4f} ")

    return acc_clean,acc_noisy,acc_g,acc_ng,acc_cn,acc_cg

def model_evaluate(config,devices, train_data, train_noise, train_g,train_label,
                   test_data, test_noise,test_g, test_label):
    train_X_clean = prepare_inputs(train_data)
    train_X_noisy = prepare_inputs(train_noise)
    train_X_g = prepare_inputs(train_g)
    train_y = np.array(train_label) if train_label is not None else np.array(train_label_ds).reshape(-1)
    train_y =  np.array(train_y).flatten()


    test_X_clean = prepare_inputs(test_data)
    test_X_noisy = prepare_inputs(test_noise)
    test_X_g = prepare_inputs(test_g)
    test_y = np.array(test_label) if test_label is not None else np.array(test_label_ds).reshape(-1)
    test_y= np.array(test_y).flatten()

    models1 = build_model(config,devices)
    _,acc_c=train_on_batch(config,models1,devices,train_X_clean,train_y,test_X_clean,test_y)
    del models1
    models2 = build_model(config, devices)
    _, acc_n = train_on_batch(config, models2, devices, train_X_noisy, train_y, test_X_noisy, test_y)
    del models2
    models3 = build_model(config, devices)
    _, acc_g = train_on_batch(config, models3, devices, train_X_g, train_y, test_X_g, test_y)
    del models3
    models4 = build_model(config, devices,concatenate=True)
    _, acc_ng = train_on_batch(config, models4, devices, np.concatenate([train_X_noisy,train_X_g],axis=-1),train_y, np.concatenate([test_X_noisy,test_X_g],axis=-1), test_y)
    del models4
    # models5 = build_model(config, devices, concatenate=True)
    # _, acc_cn = train_on_batch(config, models5, devices, np.concatenate([train_X_clean,train_X_noisy],axis=-1),train_y, np.concatenate([test_X_clean,test_X_noisy],axis=-1), test_y)
    # del models5
    # models6 = build_model(config, devices, concatenate=True)
    # _, acc_cg = train_on_batch(config, models6, devices, np.concatenate([train_X_clean,train_X_g],axis=-1),train_y, np.concatenate([test_X_clean,test_X_g],axis=-1), test_y)
    # del models6
    return acc_c.item(),acc_n.item(),acc_g.item(),acc_ng.item(),0,0

def train_on_batch(config,Models, device, train_data, train_label,
                   test_data, test_label,lr_jitter=False):
    datasetid = config["train_param"]['datasets']
    Es = config["model_param"]['Es']
    bz = config["train_param"]['bz']
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    DLalgorithm = config['train_param']["testmethod"]
    num_epochs = config[f"{DLalgorithm}"]['epochs']
    if DLalgorithm == "CCNN":
        train_data = CCNN.complex_spectrum_features(np.expand_dims(train_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
        test_data = CCNN.complex_spectrum_features(np.expand_dims(test_data, axis=1), FFT_PARAMS=[Fs, ws]).squeeze(1)
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)  # 分类标签通常用long类型
    EEGData_Train = torch.utils.data.TensorDataset(train_data.unsqueeze(1), train_label.unsqueeze(1))
    EEGData_Test = torch.utils.data.TensorDataset(test_data.unsqueeze(1), test_label.unsqueeze(1))

    # Create DataLoader for the Dataset

    train_iter = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                             drop_last=True)
    test_iter = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                            drop_last=True)

    for epoch in range(num_epochs):
        # ==================================training procedure==========================================================
        for es in range(Es):
            net = Models[es]['model']
            optimizer = Models[es]['optimizer']
            criterion = Models[es]['criterion']
            if DLalgorithm == "DDGCNN":
                lr_decay_rate = config[DLalgorithm]['lr_decay_rate']
                optim_patience = config[DLalgorithm]['optim_patience']
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                                    patience=optim_patience, verbose=True, eps=1e-08)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                                    eta_min=5e-6)
            net.train()
            sum_loss = 0.0
            sum_acc = 0.0

            for data in train_iter:
                X, y = data
                X = X.to(torch.float).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X)

                loss = criterion(y_hat, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_jitter and DLalgorithm != "DDGCNN":
                    scheduler.step()
                sum_loss += loss.item() / y.shape[0]
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

            train_loss = sum_loss / len(train_iter)
            train_acc = sum_acc / len(train_iter)
            if lr_jitter and DLalgorithm == "DDGCNN":
                scheduler.step(train_acc)

        if (epoch+1)%10 == 0:
            sum_acc = 0.0  # 把这行提前放在外层，整个 test_iter 用一次
            total_batches = 0

            for data in test_iter:
                total_batches += 1
                # ========== 获取输入 ==========
                X, y = data
                X = X.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

                # ========== 模型集成输出（平均 logits） ==========
                total_logits = 0
                for es in range(Es):
                    net = Models[es]['model']
                    net.eval()
                    with torch.no_grad():
                        logits = net(X)
                        total_logits += logits

                avg_logits = total_logits / Es
                y_pred = avg_logits.argmax(dim=-1)
                acc = (y == y_pred).float().mean()

                sum_acc += acc

            # ========== 验证集准确率 ==========
            val_acc = sum_acc / total_batches
            print(f"epoch:{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, valid_acc={val_acc:.3f}")

    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.cuda.empty_cache()
    return Models[-1]['acc_1'],val_acc.cpu().data

def build_model(config,devices,concatenate=False):
    
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''

    KLG = config['model_param']['KLG']
    DL = config['model_param']['DL']
    Nm = config["model_param"]['Nm']
    Es = config["model_param"]['Es']
    DLalgorithm = config['train_param']["testmethod"]
    datasetid = config["train_param"]["datasets"]
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of channels
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    lr = config[DLalgorithm]['lr']
    wd = config[DLalgorithm]['wd']
    initfreq = config[f"data_param{datasetid}"]["initfreq"]
    deltafreq = config[f"data_param{datasetid}"]["deltafreq"]
    Nt = int(Fs * ws)
    if concatenate:
        Nt=Nt*2
    Models = []
    acc_1 = 0
    for es in range(Es):
        if datasetid >= 2:
            DLinput = 9 * max(Nm, 1)
        else:
            DLinput = Nc * max(Nm, 1)
        if DLalgorithm == "EEGNet":
            DLnet = EEGNet.EEGNet(DLinput, Nt, Nf)
        elif DLalgorithm == "CCNN":
            DLnet = CCNN.CNN(DLinput, 220, Nf)
        elif DLalgorithm == "FBtCNN":
            DLnet = FBtCNN.tCNN(DLinput, Nt, Nf, Fs)
        elif DLalgorithm == "ConvCA":
            DLnet = ConvCA.convca(DLinput, Nt, Nf)
        elif DLalgorithm == "SSVEPformer":
            DLnet = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, FFT_PARAMS=[Fs, ws],
                                            chs_num=DLinput, class_num=Nf,
                                            dropout=0.5, resolution=deltafreq, start_freq=initfreq, end_freq=64)
            DLnet.apply(Constraint.initialize_weights)

        elif DLalgorithm == "SSVEPNet":
            DLnet = SSVEPNet.ESNet(DLinput, Nt, Nf)
            DLnet = Constraint.Spectral_Normalization(DLnet)

        elif DLalgorithm == "DDGCNN":
            bz = config[DLalgorithm]["bz"]
            norm = config[DLalgorithm]["norm"]
            act = config[DLalgorithm]["act"]
            trans_class = config[DLalgorithm]["trans_class"]
            n_filters = config[DLalgorithm]["n_filters"]
            DLnet = DDGCNN.DenseDDGCNN([bz, Nt, DLinput], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3,
                                       nclass=Nf,
                                       bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)
        model = DLnet

        if config['train_param']["smooth"] and DLalgorithm == "SSVEPNet":
            if datasetid == 2:
                criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=40)
            else:
                criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=12)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")

        if DLalgorithm == "SSVEPformer":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
        model = model.to(devices)
        Models.append({"model": model, "criterion": criterion, "optimizer": optimizer, "acc_1": acc_1})
    return Models