
# Efficient Siamese Network Based Approach for Multi-Class ECG Classification in Arrhythmia Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

> Official repository for our paper:  
> **"Efficient Siamese Network Based Approach for Multi-Class ECG Classification in Arrhythmia Detection"**  
> Accepted at 2025 Fifth International Conference on Advances in Electrical, Computing, Communications and Sustainable Technologies (ICAECT 2025).

## 📌 Abstract

Accurate ECG signal classification is vital for the early diagnosis and treatment of arrhythmias. This work proposes a **Siamese Convolutional Neural Network (SCNN)** architecture that leverages **Few-Shot Learning** and **wavelet-based scalogram transforms** to classify 17 types of cardiac arrhythmias using the MIT-BIH dataset. Our approach addresses dataset imbalance and enhances generalization, achieving a validation accuracy of **97.5%**.

---

## 🧠 Key Features

- 🧬 Few-Shot Learning with Siamese Network architecture
- 💡 Wavelet-based scalogram transformation for time-frequency representation
- 🔄 Five-stage data augmentation pipeline
- 🔍 Benchmark comparisons with 1D-CNN and AlexNet
- 📊 Achieves SOTA performance with **97.5% accuracy** on a limited dataset

---

## 📁 Project Structure

```
├── data/
│   └── mitbih/                   # Preprocessed and augmented ECG scalogram images
├── models/
│   ├── siamese_model.py         # Siamese Network architecture
│   ├── alexnet_model.py         # AlexNet benchmark
│   └── cnn_1d_model.py          # 1D CNN benchmark
├── notebooks/
│   └── analysis_and_visuals.ipynb  # Confusion matrices, ROC, PR curves
├── utils/
│   └── data_augmentation.py     # Time-warping, scaling, noise addition, etc.
├── train.py                     # Training script with CLI
├── evaluate.py                  # Evaluation script
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **MIT-BIH Arrhythmia Database**  
  - Source: [PhysioNet](https://physionet.org/content/mitdb/)
  - 1,000 ECG fragments (45 subjects)
  - 17 arrhythmia classes (multi-class classification)
- **Data Augmentation Techniques:**
  - Gaussian noise addition
  - Time shifting
  - Amplitude scaling
  - Time warping
  - Signal cleaning

---

## 🛠️ Installation

```bash
git clone git@github.com:hrshankar2002/Siamese-Convolutional-Neural-Network.git
cd Siamese-Convolutional-Neural-Network
pip install -r requirements.txt
```

---

## 🚀 Usage

### Data Retrieval
```bash
python3 get_data.py
```

### Data Augmentation

```bash
python3 data_aug.py \
  --input_folder ./path_to_data \
  --count 6000 \
  --aug_type wavelet \
  --to_aug 1
```

### Train the Siamese Network

```bash
python train.py \
  --model siamese \
  --epochs 10 \
  --batch_size 16 \
  --data_dir ./path_to_data dir 
```

### Evaluate the Trained Model

```bash
python evaluate.py \
  --model siamese \
  --weights ./path_to_h5_file \ 
  --data_dir "path/to/data dir"
```

---

## 📈 Results

| Model             | Train Acc | Val Acc | Precision | Recall | F1-Score |
|------------------|-----------|---------|-----------|--------|----------|
| 1D-CNN           | 99.87%    | 95.00%  | 95.22%    | 95.00% | 95.74%   |
| AlexNet          | 98.98%    | 97.17%  | 98.43%    | 95.40% | 96.80%   |
| **Proposed SCNN**| **99.99%**| **97.50%**| **97.56%**| **97.50%**| **97.46%** |

---

## 🧪 Experiments

- **Hardware**: NVIDIA RTX 3050 Mobile, Ryzen 7 6800H
- **Framework**: TensorFlow 2.0 + CUDA 12.4
- **Input**: Scalogram images of ECG signals
- **Optimizer**: Adam, LR = 0.001
- **Loss**: Sparse Categorical Crossentropy

---

## 📄 Citation

If you use this repository or our method, please cite:

```
@INPROCEEDINGS{10958947,
  author={P, Lubaib and J, Dhoulath Beegum and S., Harishankar V and Harikumar, Abhijith and Anilkumar, Ajul and K, Arjun P},
  booktitle={2025 Fifth International Conference on Advances in Electrical, Computing, Communication and Sustainable Technologies (ICAECT)}, 
  title={Efficient Siamese Network Based Approach for Multi-Class ECG Classification in Arrhythmia Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Hospitals;Heart beat;Arrhythmia;Computational modeling;Neural networks;Electrocardiography;Real-time systems;Convolutional neural networks;Few shot learning;Monitoring;ECG Classification;Siamese Neural Networks;Deep Learning;Arrhythmia Detection;Few-Shot Learning},
  doi={10.1109/ICAECT63952.2025.10958947}}
```

---

## 📬 Contact

For questions, drop an email:

- 📧 hrshankar2002@gmail.com

---
