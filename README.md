# SCEAF-UNet
## Usage Instructions

### 1. Data Preparation

The **Synapse (BTCV preprocessed)** and **ACDC** datasets can be obtained from the official **TransUNet** repository:

🔗 https://github.com/Beckschen/TransUNet/tree/main

Please follow the instructions provided in the TransUNet repository to download and organize the datasets properly before training or testing.

---

### 2. Environment Setup

Please create a Python environment with **Python 3.8**, and install the required dependencies by running:

```bash
pip install -r requirements.txt

```
### 3. Pretrained Weights Download
The pretrained and trained model weights can be downloaded from the following link:

🔗 https://drive.google.com/drive/my-drive

The provided weight files include:

backbone.pth: Pretrained weights for the encoder (backbone network)

epoch_30.pth: Model weights trained on the Synapse dataset

best_model.pth: Model weights trained on the ACDC dataset

Please place the downloaded weight files in the appropriate directory before running training or inference.

---
### 4. Training and Testing

To train the model on the Synapse dataset, run:
```bash
python train.py \
  --dataset Synapse \
  --max_epochs 30 \
  --base_lr 0.001 \
  --img_size 224 \
  --pretrained_path backbone.pth

```

To evaluate the trained model on the Synapse dataset, run:
```bash
python test.py \
  --dataset Synapse \
  --path_specific ckpts/epoch_30.pth

```
## Acknowledgements
This codebase partially adopts code components and auxiliary functions from **RWKV-UNet**.
We sincerely thank the authors for making their implementation publicly available.

