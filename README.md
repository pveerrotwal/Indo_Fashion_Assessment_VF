# Indo Fashion Image Classifier

## Overview
This project builds a production-style deep learning pipeline to classify Indian clothing categories using the [IndoFashion dataset](https://indofashion.github.io/). It uses transfer learning with EfficientNet-B0 pretrained on ImageNet, then fine-tunes on a curated subset created by sampling up to 500 images per class from train/val/test annotations.

## Dataset
- Source: [IndoFashion](https://indofashion.github.io/)
- Categories (15):
  - blouse
  - dhoti_pants
  - dupattas
  - gowns
  - kurta_men
  - lehenga
  - mojaris_men
  - mojaris_women
  - nehru_jacket
  - palazzo
  - petticoats
  - saree
  - sherwanis
  - women_kurta
  - leggings_and_salwars
- Subset construction:
  - Target: 500 images per class
  - Actual in this run:
    - 13 classes had >=500 samples and were used as 500 each
    - `nehru_jacket` and `palazzo` had 0 available samples in provided annotations
  - Final subset size: 6,500 images
  - 80/20 split: 5,200 train / 1,300 val

## Model Architecture
- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Classification head: `Dropout(0.3) -> Linear(1280, 15)`
- Two-phase fine-tuning:
  - Epochs 1-5: train classifier head only (frozen backbone)
  - Epochs 6-20: unfreeze backbone for full fine-tuning

## Training Details
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss with `label_smoothing=0.1`
- Mixed precision: `torch.cuda.amp`
- Early stopping: patience = 5
- Augmentations: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation

## How to Run
1. Clone this repository and enter the project folder.
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip3 install -r requirements.txt
   ```
3. Download IndoFashion manually, then prepare subset:
   ```bash
   python3 prepare_dataset.py --dataset_path /path/to/indofashion
   ```
4. Train and evaluate:
   ```bash
   python3 main.py --mode both
   ```
5. Evaluate only (using best saved checkpoint):
   ```bash
   python3 main.py --mode eval
   ```

## Results
- Best validation accuracy: **63.08%**
- Top-5 validation accuracy: **93.69%**
- Best checkpoint: epoch **5**
- Device used for this run: **CPU**

### Per-class Metrics (Validation)
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| blouse | 0.7965 | 0.9000 | 0.8451 | 100 |
| dhoti_pants | 0.5876 | 0.5700 | 0.5787 | 100 |
| dupattas | 0.5926 | 0.1600 | 0.2520 | 100 |
| gowns | 0.5057 | 0.4400 | 0.4706 | 100 |
| kurta_men | 0.7069 | 0.4100 | 0.5190 | 100 |
| lehenga | 0.6111 | 0.8800 | 0.7213 | 100 |
| mojaris_men | 0.7544 | 0.8600 | 0.8037 | 100 |
| mojaris_women | 0.7345 | 0.8300 | 0.7793 | 100 |
| nehru_jacket | 0.0000 | 0.0000 | 0.0000 | 0 |
| palazzo | 0.0000 | 0.0000 | 0.0000 | 0 |
| petticoats | 0.6907 | 0.6700 | 0.6802 | 100 |
| saree | 0.4740 | 0.7300 | 0.5748 | 100 |
| sherwanis | 0.6585 | 0.8100 | 0.7265 | 100 |
| women_kurta | 0.5072 | 0.3500 | 0.4142 | 100 |
| leggings_and_salwars | 0.5673 | 0.5900 | 0.5784 | 100 |

### Notes on This Run
- Because `nehru_jacket` and `palazzo` were absent in the provided dataset annotations, those classes had zero support in validation and therefore zero-valued metrics.
- Training reached strong performance by epoch 5; full-backbone fine-tuning from epoch 6 onward was significantly slower on CPU.

### Plots
- Training Curves: <img width="4200" height="1500" alt="image" src="https://github.com/user-attachments/assets/a9174320-6e9f-41a1-9e37-7ee18434c5ae" />

- Confusion Matrix: <img width="3900" height="3000" alt="image" src="https://github.com/user-attachments/assets/9569a3a9-dd71-41b0-a950-fdc08a2ba6ac" />

- Sample Predictions: <img width="3600" height="3600" alt="image" src="https://github.com/user-attachments/assets/58833927-0ebd-43d4-9ccd-70934c346a27" />


## Project Structure
```text
indo-fashion-classifier/
├── data/
│   ├── raw/
│   └── subset/
│       ├── train/
│       └── val/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
├── prepare_dataset.py
├── main.py
├── config.py
├── requirements.txt
└── README.md
```
