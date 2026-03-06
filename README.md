<div align="center">
<h1>DenoNet</h1>
<h3>A GAN-VAE Hybrid Generative Model for SSVEP Denoising</h3>

Zhentao He<sup>1,†</sup>, [Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1,†</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>†</sup> Z. He and Z. Wang contributed equally to this work.

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

<sup>:email:</sup> Corresponding Author

</div>

> This repository contains the implementation of our paper: **"DenoNet: A GAN-VAE Hybrid Generative Model for SSVEP Denoising"**. By combining the distribution-matching capability of generative adversarial networks (GANs) with the stable representation learning of variational autoencoders (VAEs), DenoNet learns a discriminative latent space that effectively separates noise from task-related EEG components. Extensive experiments on two public SSVEP datasets demonstrated that DenoNet consistently outperformed six baseline models and improved the decoding performance across traditional and deep learning classifiers under five noise conditions.

## Overview
**DenoNet**, a **GAN-VAE hybrid generative model** tailored for SSVEP Denoising:

- DenoNet learns a structured latent space to explicitly separate noise factors from task-related neural activities, which enables joint denoising and data augmentation, thus providing a unified solution under both low SNR and limited data conditions.
- We introduce a latent alignment constraint that enforces representation consistency between reference and denoised EEG, effectively alleviating identity-mapping pitfall and improving robustness to distribution shifts.
- Extensive experiments on two public SSVEP datasets demonstrate that DenoNet almost always outperformed six baseline denoising models, and consistently improves performance across representative traditional and deep learning classifiers under five noise conditions.
- We further provide a systematic noise sensitivity analysis for SSVEP decoding, revealing the dominant impact of spectral domain interference and highlighting the importance of frequency-aware denoising.

<div align="center">
<img width="360" height="402" alt="image" src="https://github.com/user-attachments/assets/be7a39e6-33c9-45aa-92ac-91a6db3dd5e7" />
</div>

## Baselines
Six EEG decoding models were reproduced and compared with the proposed DenoNet in this paper. DenoNet achieves the **state-of-the-art performance**. Besides, DenoNet improved the decoding performance across traditional and deep learning classifiers under five noise conditions.

<div align="center">
<img width="1225" height="863" alt="image" src="https://github.com/user-attachments/assets/6411c1a5-60de-4b07-abb3-ddc27e8e7e47" />
</div>


## Evaluation Tasks
To evaluate the effectiveness of the proposed DenoNet under different scenarios:
 
- EEG denoising: the generated clean signal is directly fed into the subsequent SSVEP decoding models, and the classification result is used to assess the denoising capability of DenoNet.
- EEG data augmentation: the generated EEG signals are combined with the original noisy signals to form an augmented training set, and the expanded dataset is then used to train the SSVEP decoding models. This setting aims to investigate whether the generated signals can enrich the training data and improve classification performance.

## Code Structure
```
DenoNet/
│
├── etc/                         # Configuration files
│   ├── config.yaml              # Training hyperparameters and experiment settings
│   └── global_config.py         # Unified configuration loader
│
├── Models/                      # Model architectures
│   ├── Generator.py             # Generator module of DenoNet (CNN, CNN+Transformer, CNN+LSTM)
│   ├── Discriminator.py         # Discriminator module of DenoNet
│   │
│   ├── CT_DCENet/               # Baseline EEG denoising model
│   ├── EEGDNet/                 # Baseline EEG denoising model
│   ├── GCTNet/                  # Baseline EEG denoising model
│   │
│   ├── DeepL/                   # Downstream classifier (baseline evaluation)
│   ├── HZTKD/                   # Downstream classifier (baseline evaluation)
│   └── KNoW/                    # Downstream classifier (baseline evaluation)
│
├── Utils/                       # Utility modules
│   ├── Trainer.py               # Core training pipeline
│   ├── dataprocess.py           # Noise injection and DataLoader construction
│   ├── EEGDataset.py            # EEG dataset reader
│   ├── test.py                  # Evaluation of denoised EEG on downstream models
│   ├── testbaseline.py          # Baseline model testing entry
│   ├── saveresult.py            # Result saving utilities
│   │
│   ├── Constraint.py            # Training constraint definitions
│   ├── LossFunction.py          # Classification loss functions with margin and smoothing
│   ├── Normalization.py         # Data normalization utilities
│   └── Script.py                # EEG preprocessing and graph-based data augmentation utilities
│
```

---

## 📄 Citation
If you find this work helpful, please stay tuned for our full paper, which will be available on arXiv soon.
We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.

