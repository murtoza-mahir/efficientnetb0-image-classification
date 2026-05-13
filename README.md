# 🧠 Image Classification with EfficientNetB0 — Transfer Learning

## 📌 Project Overview

This project implements **image classification using Transfer Learning** with **EfficientNetB0** — one of Google's most efficient and accurate convolutional neural network architectures. The model is pre-trained on ImageNet and fine-tuned on a custom dataset using a two-stage training strategy: first training the classification head with the backbone frozen, then fine-tuning the top layers for higher accuracy.

---

## 🎯 What This Project Does

- Loads a custom image dataset using `tf.keras.utils.image_dataset_from_directory`
- Applies **data augmentation** (random flip, rotation, zoom) to improve generalization
- Uses **EfficientNetB0** as a frozen feature extractor in Stage 1
- **Fine-tunes the last 30 layers** of EfficientNetB0 in Stage 2 for higher accuracy
- Evaluates final model performance on the validation set

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow 2.18 | Deep learning framework |
| Keras | Model building & training |
| EfficientNetB0 | Pre-trained backbone (ImageNet) |
| Google Colab | Training environment |
| Matplotlib | Training curve visualization |

---

## 🏗️ Model Architecture

```
Input (224x224x3)
    ↓
EfficientNetB0 Preprocessing
    ↓
EfficientNetB0 Base (frozen → then fine-tuned last 30 layers)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.25)
    ↓
Dense (num_classes, softmax)
```

---

## 🔄 Training Strategy

**Stage 1 — Feature Extraction (Frozen Backbone)**
- EfficientNetB0 backbone fully frozen
- Only classification head trained
- Optimizer: Adam (lr=1e-4)
- Epochs: 10 (with EarlyStopping, patience=3)

**Stage 2 — Fine-Tuning (Top 30 Layers Unfrozen)**
- Last 30 layers of EfficientNetB0 unfrozen (BatchNorm layers kept frozen)
- Optimizer: Adam (lr=1e-5)
- Callbacks: ModelCheckpoint (saves best model), ReduceLROnPlateau

---

## 📂 Repository Structure

```
efficientnetb0-image-classification/
│
├── Copy_of_efficientnet_b0_project.ipynb   # Main notebook
├── README.md                                # Project documentation
└── dataset/                                 # Your dataset folder (not included)
    ├── class_1/
    ├── class_2/
    └── class_n/
```

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Upload your dataset to Google Drive in a folder with sub-folders per class
3. Update `data_dir` path in the notebook to point to your dataset
4. Run all cells from top to bottom
5. Best model saved as `best_efficientnet.h5`

---

## 💡 Key Concepts Demonstrated

- Transfer Learning with pre-trained ImageNet weights
- Two-stage fine-tuning strategy
- Data augmentation for regularization
- EarlyStopping and ReduceLROnPlateau callbacks
- Model checkpointing for best weights

---

## 👤 Author

**Md Murtoza Mahir**
Data Analyst | ML Engineer | Power BI | Python | SQL
📧 murtozamahir.info@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/murtoza-mahir)
🌐 [Portfolio](https://murtoza-mahir.github.io)
