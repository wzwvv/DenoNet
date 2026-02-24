from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
import time
from Models import Discriminator,Generator,VAE
from Models.GCTNet.models.GCTNet import Discriminator as GCTNet_D
from Utils.dataprocess import add_noise
from etc.global_config import config
import torch
import random
from torch.nn import functional as F
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
def train_GCTNet(
    eeg_train_dataloader,
    eeg_test_dataloader,
    model,          # generator
    epochs,
    device
):
    w_f = 0.05
    w_c = 0.05
    w_cls=0.05
    datasetid = config["train_param"]['datasets']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    # ======================
    # Discriminator
    # ======================
    model_d = GCTNet_D(Nf=Nf).to(device)
    model_d.apply(weights_init)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8
    )
    optimizer_D = torch.optim.Adam(model_d.parameters(), lr=0.0001)

    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        model_d.train()
        losses = []

        for x_prime, X, y in eeg_train_dataloader:
            # ========= 数据 =========
            X = X.float().squeeze().to(device)          # contaminated
            x_prime = x_prime.float().squeeze().to(device)  # clean
            y=torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
            # ===== 多通道 → 单通道 =====
            if X.dim() == 3:
                B, C, L = X.shape
                X = X.view(B * C, L).unsqueeze(dim=1)
                x_prime = x_prime.view(B * C, L)
                y = y.repeat_interleave(C)
                
            p_t = model(X).view(X.shape[0], -1)

            fake_y, fake_cls, _, _, _ = model_d(p_t.unsqueeze(1))
            real_y, real_cls, _, _, _ = model_d(x_prime.unsqueeze(1))

            d_loss = (
                0.5 * torch.mean(fake_y ** 2) +
                0.5 * torch.mean((real_y - 1) ** 2)
            )

            # 🔹 新增：判别器分类损失
            d_cls_loss = ce(real_cls, y) + ce(fake_cls, y)
            d_loss = d_loss + w_c * d_cls_loss

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # =========================
            # 2. 训练 Generator
            # =========================
            p_t = model(X).view(X.shape[0], -1)

            fake_y, fake_cls, _, fake_feature2, _ = model_d(p_t.unsqueeze(1))
            _, real_cls, _, true_feature2, _ = model_d(x_prime.unsqueeze(1))

            # 原始去噪损失
            loss_mse = denoise_loss_mse(p_t, x_prime)

            # feature loss
            loss_feat = denoise_loss_mse(fake_feature2, true_feature2)

            # 分类损失
            loss_cls = ce(fake_cls, y) 
            g_loss = loss_mse + w_f * loss_feat + w_c * (torch.mean((fake_y - 1) ** 2))  + w_cls* loss_cls

            optimizer.zero_grad()
            optimizer_D.zero_grad()
            g_loss.backward()
            optimizer.step()

            losses.append(g_loss.detach())

        print(
            f"\rEpoch [{epoch+1}/{epochs}] "
            f"Train loss: {torch.stack(losses).mean().item():.4f}",
            end=""
        )

    # =========================
    # 测试：只算分类准确率
    # =========================
    model.eval()
    model_d.eval()

    correct_fake, correct_real, total = 0, 0, 0

    with torch.no_grad():
        for x_prime, X, y in eeg_test_dataloader:
            X = X.float().squeeze().to(device)          # contaminated
            x_prime = x_prime.float().squeeze().to(device)  # clean
            y=torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

            if X.dim() == 3:
                B, C, L = X.shape
                X = X.view(B * C, L).unsqueeze(dim=1)
                x_prime = x_prime.view(B * C, L)
                y = y.repeat_interleave(C)

            p_t = model(X).view(X.shape[0], -1)

            fake_y, fake_cls, _, _, _ = model_d(p_t.unsqueeze(1))
            real_y, real_cls, _, _, _ = model_d(x_prime.unsqueeze(1))

            fake_pred = fake_cls.argmax(dim=1)
            real_pred = real_cls.argmax(dim=1)

            correct_fake += (fake_pred == y).sum().item()
            correct_real += (real_pred == y).sum().item()
            total += y.size(0)

    test_acc_fake_cls = correct_fake / total
    test_acc_real_cls = correct_real / total

    print(
        f"Test fake-cls acc: {test_acc_fake_cls:.4f}, "
        f"real-cls acc: {test_acc_real_cls:.4f}"
    )

    return model, model_d, test_acc_fake_cls, test_acc_real_cls
def train_gan(eeg_train_dataloader, eeg_test_dataloader, device,basegenerator=None):
    datasetid = config["train_param"]['datasets']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    epochs = config["train_param"]["epochs"]
    Nm = config["model_param"]["Nm"]
    Nt = int(Fs * ws)
    #Nc = 9
    if basegenerator == None:
        generator = Generator.EEGDenoiseGeneratorv2(Nc*Nm,Nt).to(device)  
    else:
        generator = basegenerator.to(device) 
    discriminator = Discriminator.Spectral_Normalization(Discriminator.ESNet(Nc*Nm,Nt,Nf)).to(device)
    #discriminator = Discriminator.ESNetv2(Nc*Nm,Nt,Nf,device=device)
    #print(generator,discriminator)
    wd_D = config["gan_model_param"]["wd_D"]
    wd_G = config["gan_model_param"]["wd_G"]
    lr_G = config["gan_model_param"]["lr_G"]
    lr_D = config["gan_model_param"]["lr_D"]
    lambda_G = config["gan_model_param"]["lambda_G"]
    lambda_D = config["gan_model_param"]["lambda_D"]
    lambda_vae = config["gan_model_param"]["lambda_vae"]
    lambda_kl = config["gan_model_param"]["lambda_kl"]
    lr_jitter = config["gan_model_param"]["lr_jitter"]

    noisetype = config["train_param"]['noisetype']
    noise_ratio = config["train_param"]['noise_ratio']
    snr_db = config["train_param"]['snr_db']
    generator.train()
    discriminator.train()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G,weight_decay=wd_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D,weight_decay=wd_D)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs * len(eeg_train_dataloader),
                                                                        eta_min=5e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs * len(eeg_train_dataloader),
                                                                        eta_min=5e-6)
    criterion_ce = nn.CrossEntropyLoss()
    log = lambda x: torch.log(x + 1e-8)  # 避免 log(0)

    for epoch in range(epochs):
        discriminator.train()
        generator.train()
        for data in eeg_train_dataloader:
            x_prime,X, y = data
            X = X.type(torch.FloatTensor).to(device)
            x_prime = x_prime.type(torch.FloatTensor).to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
            # ========================
            # 1. 训练 Discriminator
            # ========================
            optimizer_D.zero_grad()

            with torch.no_grad():
                _,_,_,_,gen_x = generator(X)     # 生成去噪EEG：G(X)
                
                meanref1,log_varianceref1,meanref2,log_varianceref2,_=generator(x_prime)

            # 判别器对真实信号的判断
            d_real = discriminator(x_prime.detach())
            d_real_prob = d_real['adv']
            d_real_cls = d_real['cls']
            # 判别器对生成信号的判断
            d_fake = discriminator(gen_x.detach())
            d_fake_prob = d_fake['adv']
            d_fake_cls = d_fake['cls']

            # Discriminator Loss
            if basegenerator == None:
                L_Dadv = -log(d_real_prob).mean() - log(1 - d_fake_prob).mean()
            else:
                L_Dadv = 0
            L_feat = -log(d_real_cls[range(len(y)), y]).mean() - log(d_fake_cls[range(len(y)), y]).mean()
            
            if lambda_D == 0:
                loss_D = L_Dadv
            else:
                loss_D = L_Dadv+lambda_D * L_feat
            loss_D.backward()
            optimizer_D.step()
            if lr_jitter:
                scheduler_D.step()

            # ========================
            # 2. 训练 Generator
            # ========================
            optimizer_G.zero_grad()

            mean1,log_variance1,mean2,log_variance2,gen_x = generator(X)  # 重新生成，避免 detach
            

            d_gen = discriminator(gen_x)
            d_gen_prob = d_gen['adv']
            d_gen_cls = d_gen['cls']

            # Generator Loss
            L_ca = criterion_ce(d_gen_cls, y)  # 分类损失
            if basegenerator == None:
                L_Gadv = -log(d_gen_prob).mean()        # 骗过判别器的对抗损失
            else:
                L_Gadv =0
            recon_loss = F.l1_loss(gen_x, x_prime, reduction='mean')
            #kl_loss = -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())
            var1 = torch.exp(log_variance1)
            varref1 = torch.exp(log_varianceref1)
            var2 = torch.exp(log_variance2)
            varref2 = torch.exp(log_varianceref2)
            
            kl_loss1 = 0.5 * torch.mean(log_varianceref1 - log_variance1 + (var1 + (mean1 - meanref1).pow(2)) / varref1- 1)
            kl_loss2 = 0.5 * torch.mean(log_varianceref2 - log_variance2 + (var2 + (mean2 - meanref2).pow(2)) / varref2- 1)
            kl_loss = kl_loss1+kl_loss2
            loss_G = L_Gadv
            if lambda_G != 0 :
                loss_G = loss_G +lambda_G * L_ca
            if lambda_vae != 0 :
                loss_G = loss_G + lambda_vae*recon_loss
            if ( lambda_kl != 0) and (basegenerator == None):
                loss_G = loss_G + lambda_kl*kl_loss
            #print(L_Gadv,lambda_G *L_ca,lambda_vae*recon_loss,lambda_kl*kl_loss)

            loss_G.backward()
            optimizer_G.step()
            if lr_jitter:
                scheduler_G.step()
        if (epoch+1)%10 == 0:
            sum_acc_adv = 0.0  # 把这行提前放在外层，整个 test_iter 用一次
            sum_acc_fake_cls = 0.0
            sum_acc_real_cls = 0.0
            discriminator.eval()
            dataset_length = 0
            generator.eval()
            for data in eeg_test_dataloader:
                # ========== 获取输入 ==========
                x_prime,X,y = data
                X = X.type(torch.FloatTensor).to(device)
                x_prime = x_prime.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                dataset_length+=len(y)
                _,_,_,_,output = generator(X)
                # ========== 模型集成输出（平均 logits） ==========
                d_fake = discriminator(output.detach())
                d_fake_prob = d_fake['adv']
                d_fake_cls = d_fake['cls']

                d_real = discriminator(x_prime.detach())
                d_real_prob = d_real['adv']
                d_real_cls = d_real['cls']

                d_real_correct = (d_real_prob >= 0.5).float()
                d_fake_correct = (d_fake_prob < 0.5).float()
                acc1 = d_real_correct.sum() + d_fake_correct.sum()
                sum_acc_adv += acc1

                fake_preds = d_fake_cls.argmax(dim=1)
                sum_acc_fake_cls =sum_acc_fake_cls+ (fake_preds == y).float().sum()
                real_preds = d_real_cls.argmax(dim=1)
                sum_acc_real_cls = sum_acc_real_cls + (real_preds == y).float().sum()
                # ========== 验证集准确率 ==========
            val_acc_adv = sum_acc_adv / dataset_length/2
            val_acc_fake_cls= sum_acc_fake_cls / dataset_length
            val_acc_real_cls = sum_acc_real_cls / dataset_length
            print(f"Epoch {epoch+1}/{epochs} | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f} | valid_acc_adv :{val_acc_adv:.3f}| valid_acc_fake_cls :{val_acc_fake_cls:.3f} | valid_acc_real_cls :{val_acc_real_cls:.3f}")

    print(
            f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc= {val_acc_adv:.3f},{val_acc_fake_cls:.3f},{val_acc_real_cls:.3f}')

    return val_acc_fake_cls,val_acc_real_cls,generator,discriminator
def cal_col(pred_list,i): # cal q_(h)
    # H = len(pred_list)
    # y_consensus = torch.stack(pred_list,dim=1) #[N,H,L]
    # y_consensus[:,i,:] = torch.zeros_like(y_consensus[:,i,:])
    # y_consensus =  y_consensus.sum(dim=1) # [N,L]
    # y_consensus = y_consensus / (H-1)
    # return y_consensus.detach()
    # choice_list = pred_list[0:i] + pred_list[i+1:]
    # return random.choice(choice_list).detach()
    # j = 1 + 4*(i//2) - i
    # return pred_list[j]
    choice_list = pred_list[0:i] + pred_list[i + 1:]
    choice = torch.stack(choice_list,dim=1) #[N,N-1,L]
    choice = choice.mean(dim=1) #[N,L]
    choice_list.append(choice)
    col = random.choice(choice_list).detach()
    return col
def collaborative_loss(i,pred_list,y_true,d_f,beta_height=0.05):
    '''

    :param y_pred:
    :param y_consensus:
    :param y_true:
    :param d_f:  loss function
    :param lamda:
    :param p:
    :return:
    '''
    # print(beta_height)
    pred = pred_list[i] # eeg_d_i
    l_target = d_f(pred,y_true)
    col = cal_col(pred_list,i) # eeg_col
    lamda = random.uniform(0,beta_height)
    l_cor = d_f(pred, col)
    loss = l_target + lamda * l_cor

    return loss
def train_ct_dcenet_with_cls(
    eeg_train_dataloader,
    eeg_test_dataloader,
    device,
    model,
    epochs_stage1=50,
    epochs_stage2=50,
    beta_height=0.05,
    lr=1e-3,
    lr_D=1e-4,
    wd_D=0.0,
    betas=(0.9, 0.99),
    eps=1e-8,
    lambda_cls=1.0,
):
    """
    CT-DCENet + supervised classification loss (cls head only)

    - Stage 1: collaborative unsupervised + cls constraint
    - Stage 2: denoise MSE + cls constraint
    """
    datasetid = config["train_param"]['datasets']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    Nm = config["model_param"]["Nm"]
    Nt = int(Fs * ws)
    model = model.to(device)

    # ========= 判别器（只用 cls 头） =========
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(Nc * Nm, Nt, Nf)
    ).to(device)

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr_D,
        weight_decay=wd_D
    )

    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D,
        T_max=(epochs_stage1 + epochs_stage2) * len(eeg_train_dataloader),
        eta_min=5e-6
    )

    criterion_ce = nn.CrossEntropyLoss()

    # ============================================================
    # Stage 1: Individual Learners (collaborative + cls)
    # ============================================================
    print("\n==== Stage 1: Training Individual Learners + CLS ====")

    for name, param in model.named_parameters():
        if name.startswith(('sl', 'supplement', 'beta')):
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=betas, eps=eps
    )

    for epoch in range(epochs_stage1):
        model.train()
        discriminator.train()

        epoch_loss = 0.0

        for x_prime, X, y in eeg_train_dataloader:
            # ===== 数据 =====
            X = X.float().squeeze().to(device)
            x_prime = x_prime.float().squeeze().to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

            # ===== 多通道 → 单通道 =====
            if X.dim() == 3:
                B, C, L = X.shape
                X = X.view(B * C, L)
                x_prime = x_prime.view(B * C, L)

            # ===== 标准化 =====
            std = X.std(dim=1, keepdim=True) + 1e-6
            Xn = X / std
            Yn = x_prime / std

            # ===== Generator forward =====
            fl_out_list = model(Xn)

            # ---- 原协同无监督损失 ----
            loss_unsup = 0.0
            for i in range(len(fl_out_list)):
                loss_unsup += collaborative_loss(
                    i, fl_out_list, Yn,
                    d_f=F.l1_loss if i % 2 == 0 else F.mse_loss,
                    beta_height=beta_height
                )

            # ---- 生成信号（ensemble）----
            gen_x = torch.mean(torch.stack(fl_out_list), dim=0)

            # =================================================
            # 判别器：只训练 cls 头
            # =================================================
            d_real = discriminator(Yn.view(B , C, L).unsqueeze(1))
            d_fake = discriminator(gen_x.view(B , C, L).detach().unsqueeze(1))

            loss_D_cls = (
                criterion_ce(d_real['cls'], y) +
                criterion_ce(d_fake['cls'], y)
            )

            optimizer_D.zero_grad()
            loss_D_cls.backward()
            optimizer_D.step()
            scheduler_D.step()

            # =================================================
            # Generator：cls 约束（不 detach）
            # =================================================
            d_fake_for_G = discriminator(gen_x.view(B , C, L).unsqueeze(1))
            loss_G_cls = criterion_ce(d_fake_for_G['cls'], y)

            loss_G = loss_unsup + lambda_cls * loss_G_cls

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

            epoch_loss += loss_G.item()
        print(
            f"\r[Stage1][Epoch {epoch+1}/{epochs_stage1}] "
            f"Loss: {epoch_loss / len(eeg_train_dataloader):.4f}",
            end=""
        )

    # ============================================================
    # Stage 2: Ensemble + Supplement (MSE + cls)
    # ============================================================
    print("\n==== Stage 2: Training Ensemble & Supplement + CLS ====")

    for name, param in model.named_parameters():
        if name.startswith(('sl', 'supplement', 'beta')):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=betas, eps=eps
    )

    for epoch in range(epochs_stage2):
        model.train()
        discriminator.train()

        epoch_loss = 0.0

        for x_prime, X, y in eeg_train_dataloader:
            X = X.float().squeeze().to(device)
            x_prime = x_prime.float().squeeze().to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

            if X.dim() == 3:
                B, C, L = X.shape
                X = X.view(B * C, L)
                x_prime = x_prime.view(B * C, L)
                

            std = X.std(dim=1, keepdim=True) + 1e-6
            Xn = X / std
            Yn = x_prime / std

            # ===== 去噪 =====
            pred = model.denoise(Xn)

            loss_mse = F.mse_loss(pred, Yn)

            # ===== 分类约束 =====
            d_fake = discriminator(pred.view(B , C, L).unsqueeze(1))
            loss_cls = criterion_ce(d_fake['cls'], y)

            loss = loss_mse + lambda_cls * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"\r[Stage2][Epoch {epoch+1}/{epochs_stage2}] "
            f"Loss: {epoch_loss / len(eeg_train_dataloader):.4f}",
            end=""
        )
    print()

    return model, discriminator
def denoise_loss_mse(denoise, clean):      
  criterion = nn.MSELoss()
  loss = criterion(denoise, clean)
  return torch.mean(torch.mean(loss))
def Train_EEGDNet(
    eeg_train_dataloader,
    eeg_test_dataloader,
    device,
    model,
    epochs=10000,
    lr=5e-4,
    lr_D=1e-4,
    wd_D=0.0,
    betas=(0.5, 0.9),
    eps=1e-8,
    lambda_cls=1.0,
):
    """
    完全忠于原始 EEGDNet / DeT 训练流程
    + 仅引入 判别器 cls 头 的监督信号

    - 单阶段
    - MSE 去噪损失
    - 无标准化
    - 无 ensemble
    - 判别器只用 cls，不用 adv
    """

    # ========= 数据参数 =========
    datasetid = config["train_param"]["datasets"]
    Nf = config[f"data_param{datasetid}"]["Nf"]
    Nc = config[f"data_param{datasetid}"]["Nc"]
    Fs = config[f"data_param{datasetid}"]["Fs"]
    ws = config["train_param"]["ws"]
    Nm = config["model_param"]["Nm"]
    Nt = int(Fs * ws)

    model = model.to(device)

    # ========= 判别器（只用 cls 头） =========
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(Nc * Nm, Nt, Nf)
    ).to(device)

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr_D,
        weight_decay=wd_D
    )

    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D,
        T_max=epochs * len(eeg_train_dataloader),
        eta_min=5e-6
    )

    criterion_ce = nn.CrossEntropyLoss()

    optimizer_G = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps
    )

    # ==========================================================
    # Training
    # ==========================================================
    for epoch in range(epochs):
        model.train()
        discriminator.train()

        epoch_loss = 0.0

        for x_clean, x_noisy, y in eeg_train_dataloader:
            # ========= 数据 =========
            x_clean = x_clean.float().to(device)
            x_noisy = x_noisy.float().to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.long).to(device)

            # ===== [B,1,C,T] → [B,C,T] =====
            if x_noisy.dim() == 4:
                x_noisy = x_noisy.squeeze(1)
                x_clean = x_clean.squeeze(1)

            # ===== [B,C,T] → [B*C,T] =====
            if x_noisy.dim() == 3:
                B, C, T = x_noisy.shape
                x_noisy = x_noisy.view(B * C, T)
                x_clean = x_clean.view(B * C, T)
            else:
                B = x_noisy.shape[0]
                C = 1
                T = x_noisy.shape[1]

            # ==================================================
            # Generator forward（EEGDNet 原始）
            # ==================================================
            pred = model(x_noisy)

            # ==================================================
            # 判别器：cls 监督（先训 D）
            # ==================================================
            with torch.no_grad():
                gen_x_detach = pred.detach()

            d_real = discriminator(
                x_clean.view(B, C, T).unsqueeze(1)
            )
            d_fake = discriminator(
                gen_x_detach.view(B, C, T).unsqueeze(1)
            )

            loss_D_cls = (
                criterion_ce(d_real["cls"], y) +
                criterion_ce(d_fake["cls"], y)
            )

            optimizer_D.zero_grad()
            loss_D_cls.backward()
            optimizer_D.step()
            scheduler_D.step()

            # ==================================================
            # Generator loss
            # ==================================================
            loss_mse = F.mse_loss(pred, x_clean)

            d_fake_for_G = discriminator(
                pred.view(B, C, T).unsqueeze(1)
            )
            loss_G_cls = criterion_ce(d_fake_for_G["cls"], y)

            loss_G = loss_mse + lambda_cls * loss_G_cls

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            epoch_loss += loss_G.item()

        print(
            f"\r[EEGDNet+CLS][Epoch {epoch+1}/{epochs}] "
            f"Loss: {epoch_loss / len(eeg_train_dataloader):.6f}",
            end=""
        )

    print("\nTraining finished.")
    return model

def vae_loss(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=0):
    """
    VAE损失函数
    recon_x: 重构的输出
    x: 原始输入
    mu: 潜在分布的均值
    logvar: 潜在分布的对数方差
    """
    # 重构损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL散度损失
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # 总损失
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
def train_vae(eeg_train_dataloader, eeg_test_dataloader, device,pre_train=False):
    datasetid = config["train_param"]['datasets']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    epochs = config["train_param"]["epochs"]
    Nm = config["model_param"]["Nm"]
    Nt = int(Fs * ws)
    # Nc = 9
    #generator = VAE.VAE(Nc * Nm, Nt).to(device)  # 你自己的生成器类
    generator=VAE.VAE(Nc * Nm, Nt).to(device)  # 你自己的生成器类
    discriminator = Discriminator.Spectral_Normalization(Discriminator.ESNet(Nc * Nm, Nt, Nf)).to(device)
    # print(generator,discriminator)
    wd_D = config["gan_model_param"]["wd_D"]
    wd_G = config["gan_model_param"]["wd_G"]
    lr_G = config["gan_model_param"]["lr_G"]
    lr_D = config["gan_model_param"]["lr_D"]
    lambda_G = config["gan_model_param"]["lambda_G"]
    lambda_D = config["gan_model_param"]["lambda_D"]
    lambda_vae = config["gan_model_param"]["lambda_vae"]
    lr_jitter = config["gan_model_param"]["lr_jitter"]
    if pre_train:
        generator.load_state_dict(torch.load("generator.pth"))
        discriminator.load_state_dict(torch.load("discriminator.pth"))
    generator.train()
    discriminator.train()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, weight_decay=wd_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, weight_decay=wd_D)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs * len(eeg_train_dataloader),
                                                             eta_min=5e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs * len(eeg_train_dataloader),
                                                             eta_min=5e-6)
    criterion_ce = nn.CrossEntropyLoss()
    log = lambda x: torch.log(x + 1e-8)  # 避免 log(0)

    for epoch in range(epochs):
        discriminator.train()
        generator.train()
        for data in eeg_train_dataloader:
            x_prime, X, y = data
            X = X.type(torch.FloatTensor).to(device)
            x_prime = x_prime.type(torch.FloatTensor).to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

            # ========================
            # 1. 训练 Discriminator
            # ========================
            optimizer_D.zero_grad()

            with torch.no_grad():
                gen_x,_,_ = generator(X)  # 生成去噪EEG：G(X)

            # 判别器对真实信号的判断
            d_real = discriminator(x_prime.detach())
            d_real_prob = d_real['adv']
            d_real_cls = d_real['cls']
            # 判别器对生成信号的判断
            d_fake = discriminator(gen_x.detach())
            d_fake_prob = d_fake['adv']
            d_fake_cls = d_fake['cls']

            # Discriminator Loss
            L_Dadv = -log(d_real_prob).mean() - log(1 - d_fake_prob).mean()
            L_feat = -log(d_real_cls[range(len(y)), y]).mean() - log(d_fake_cls[range(len(y)), y]).mean()
            loss_D = L_feat + lambda_D * L_Dadv

            loss_D.backward()
            optimizer_D.step()
            if lr_jitter:
                scheduler_D.step()

            # ========================
            # 2. 训练 Generator
            # ========================
            optimizer_G.zero_grad()

            gen_x,mu, logvar= generator(X)  # 重新生成，避免 detach
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(gen_x, x_prime, mu, logvar)

            d_gen = discriminator(gen_x)
            d_gen_prob = d_gen['adv']
            d_gen_cls = d_gen['cls']

            # Generator Loss
            L_ca = criterion_ce(d_gen_cls, y)  # 分类损失
            L_Gadv = -log(d_gen_prob).mean()  # 骗过判别器的对抗损失

            loss_G = L_ca + lambda_G * L_Gadv + loss*lambda_vae
            loss_G.backward()
            optimizer_G.step()
            if lr_jitter:
                scheduler_G.step()
        if (epoch + 1) % 10 == 0:
            sum_acc_adv = 0.0  # 把这行提前放在外层，整个 test_iter 用一次
            sum_acc_fake_cls = 0.0
            sum_acc_real_cls = 0.0
            discriminator.eval()
            dataset_length = 0
            generator.eval()
            for data in eeg_test_dataloader:
                # ========== 获取输入 ==========
                x_prime, X, y = data
                X = X.type(torch.FloatTensor).to(device)
                x_prime = x_prime.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                dataset_length += len(y)
                output,_,_ = generator(X)
                # ========== 模型集成输出（平均 logits） ==========
                d_fake = discriminator(output.detach())
                d_fake_prob = d_fake['adv']
                d_fake_cls = d_fake['cls']

                d_real = discriminator(x_prime.detach())
                d_real_prob = d_real['adv']
                d_real_cls = d_real['cls']

                d_real_correct = (d_real_prob >= 0.5).float()
                d_fake_correct = (d_fake_prob < 0.5).float()
                acc1 = d_real_correct.sum() + d_fake_correct.sum()
                sum_acc_adv += acc1

                fake_preds = d_fake_cls.argmax(dim=1)
                sum_acc_fake_cls = sum_acc_fake_cls + (fake_preds == y).float().sum()
                real_preds = d_real_cls.argmax(dim=1)
                sum_acc_real_cls = sum_acc_real_cls + (real_preds == y).float().sum()
                # ========== 验证集准确率 ==========
            val_acc_adv = sum_acc_adv / dataset_length / 2
            val_acc_fake_cls = sum_acc_fake_cls / dataset_length
            val_acc_real_cls = sum_acc_real_cls / dataset_length
            print(
                f"Epoch {epoch + 1}/{epochs} | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f} | valid_acc_adv :{val_acc_adv:.3f}| valid_acc_fake_cls :{val_acc_fake_cls:.3f} | valid_acc_real_cls :{val_acc_real_cls:.3f}")

            torch.save(generator.state_dict(), "generator.pth")
            torch.save(discriminator.state_dict(), "discriminator.pth")
    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc= {val_acc_adv:.3f},{val_acc_fake_cls:.3f},{val_acc_real_cls:.3f}')

    return val_acc_fake_cls, val_acc_real_cls, generator, discriminator
def train_vae_new(eeg_train_dataloader, eeg_test_dataloader, device,pre_train=False):
    datasetid = config["train_param"]['datasets']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    epochs = config["train_param"]["epochs"]
    Nm = config["model_param"]["Nm"]
    Nt = int(Fs * ws)
    # Nc = 9
    generator = VAE.VAE(Nc * Nm, Nt).to(device)  # 你自己的生成器类
    wd = config["vae_model_param"]["wd"]
    lr = config["vae_model_param"]["lr"]
    lambda_vae = config["vae_model_param"]["lambda_vae"]
    lr_jitter = config["vae_model_param"]["lr_jitter"]
    if pre_train:
        generator.load_state_dict(torch.load("generator_vae.pth"))
    generator.train()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=wd)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs * len(eeg_train_dataloader),
                                                             eta_min=5e-6)

    for epoch in range(epochs):
        generator.train()
        for data in eeg_train_dataloader:
            x_prime, X, y = data
            X = X.type(torch.FloatTensor).to(device)
            x_prime = x_prime.type(torch.FloatTensor).to(device)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)

            optimizer_G.zero_grad()

            gen_x,mu, logvar= generator(X)  # 重新生成，避免 detach
            # 计算损失
            loss_G, recon_loss, kl_loss = vae_loss(gen_x, x_prime, mu, logvar,kl_weight=lambda_vae)
            loss_G.backward()
            optimizer_G.step()
            if lr_jitter:
                scheduler_G.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | G Loss: {loss_G.item():.4f} ")
            torch.save(generator.state_dict(), "generator_vae.pth")
    print(f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_loss= {loss_G:.3f}')

    return generator


if __name__ == '__main__':
    # 初始化模型结构（和训练时结构完全一致）
    datasetid = config["train_param"]['datasets']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Nf = config[f"data_param{datasetid}"]["Nf"]  # number of target frequency
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency
    Fs = config[f"data_param{datasetid}"]["Fs"]  # Sample Frequency
    ws = config[f"train_param"]['ws']
    epochs = config["train_param"]["epochs"]
    Nm = config["gan_model_param"]["Nm"]
    Nt = int(Fs * ws)
    # Nc = 9
    generator = Generator.EEGDenoiseGenerator(Nc * Nm, Nt).to('cuda')  # 你自己的生成器类
    discriminator = Discriminator.Spectral_Normalization(Discriminator.ESNet(Nc * Nm, Nt, Nf)).to('cuda')

    # 加载参数
    generator.load_state_dict(torch.load("generator.pth"))
    discriminator.load_state_dict(torch.load("discriminator.pth"))

    # 如果是 GPU 训练的模型，CPU 加载时添加 map_location
    # torch.load("generator.pth", map_location=torch.device("cpu"))



