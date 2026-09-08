<div align="center">
<h1>DenoFlow</h1>
<h3>Flow Matching for SSVEP Denoising under Real Physiological Artifacts</h3>

Zhentao He<sup>1,†</sup>, [Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1,†</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>†</sup> Z. He and Z. Wang contributed equally to this work.

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

<sup>:email:</sup> Corresponding Author

</div>

> This repository contains the implementation of our paper: **"DenoFlow: Flow Matching for SSVEP Denoising under Real Physiological Artifacts"**. Although recent denoising approaches have shown promise, they are fitted without paired ground truth, can settle on reproducing their input, and are optimized on waveform distance alone, which says nothing about whether the output stays decodable. To address these issues, we propose DenoFlow, which casts SSVEP denoising as transport: instead of learning a direct map from a contaminated trial to a clean one, a field network regresses the velocity of the straight path between them, following the rectified-flow formulation, and denoising integrates that field forward from the observation. The field network is an encoder-decoder that sees the contaminated trial at every layer and the path position at its bottleneck, and a classifier trained alongside it supervises the integrated output. Because the observation itself is both the conditioning input and the starting point of the integration, the model never generates a trial from noise, and training reduces to regression, removing the adversarial min-max game. To obtain paired data on datasets with no ground truth, artifacts are injected from recorded EMG and EOG epochs under a controlled signal-to-noise target. Experiments on two public SSVEP datasets show that DenoFlow improves both signal fidelity and downstream decoding accuracy over competing denoisers.

## Overview
**DenoFlow**, a **flow matching generative model** tailored for SSVEP Denoising under real physiological artifacts:

## Baselines
Six EEG decoding models were reproduced and compared with the proposed DenoFlow in this paper. DenoFlow achieves the **state-of-the-art performance**. Besides, DenoFlow improved the decoding performance across traditional and deep learning classifiers under five noise conditions.

<div align="center">
<img width="1225" height="863" alt="image" src="https://github.com/user-attachments/assets/6411c1a5-60de-4b07-abb3-ddc27e8e7e47" />
</div>


## Evaluation Tasks
To evaluate the effectiveness of the proposed DenoFlow under different scenarios:
 
- EEG denoising: the generated clean signal is directly fed into the subsequent SSVEP decoding models, and the classification result is used to assess the denoising capability of DenoFlow.
- EEG data augmentation: the generated EEG signals are combined with the original noisy signals to form an augmented training set, and the expanded dataset is then used to train the SSVEP decoding models. This setting aims to investigate whether the generated signals can enrich the training data and improve classification performance.

## Code Structure
```
DenoFlow/
│
├── etc/                         # Configuration files
│   ├── config.yaml              # Training hyperparameters and experiment settings
│   └── global_config.py         # Unified configuration loader
│
├── Models/                      # Model architectures
│   ├── Generator.py             # Generator module of DenoFlow (CNN, CNN+Transformer, CNN+LSTM)
│   ├── Discriminator.py         # Discriminator module of DenoFlow
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

