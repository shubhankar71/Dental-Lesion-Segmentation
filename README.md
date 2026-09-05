# 🦷 Dental X-Ray Lesion Segmentation

A deep learning-based medical image segmentation project for detecting and segmenting dental cavities/lesions from **Orthopantomogram (OPG) panoramic X-ray images**.

The project uses **PyTorch** and **Segmentation Models PyTorch (SMP)** to train a semantic segmentation model capable of identifying lesion regions at the pixel level.

---

## 📌 Project Overview

Dental panoramic X-rays contain complex structures and relatively small lesion regions, making automated lesion segmentation challenging due to severe **foreground-background class imbalance**.

This project focuses on:

* Exploring and analyzing dental X-ray datasets
* Matching X-ray images with their corresponding segmentation masks
* Handling inconsistent image/mask naming conventions
* Training a deep learning segmentation model
* Addressing severe class imbalance
* Applying mask post-processing
* Visualizing predictions against ground-truth masks
* Evaluating lesion detection performance

---

## 🗂️ Dataset

The notebook works with multiple dental X-ray and segmentation-mask sources:

* **TUFTS Dental Database** https://www.kaggle.com/datasets/tommyngx/the-tufts-dental-database-2022
* **Children's Dental Panoramic Radiographs Dataset**  https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset

  * Training images
  * Test images
  * Supplemental images

The current EDA successfully matched:

| Dataset Source        | Images | Masks | Matched |
| --------------------- | -----: | ----: | ------: |
| TUFTS                 |   1000 |  1000 |    1000 |
| Children Train        |     70 |    70 |       0 |
| Children Test         |     30 |    30 |       0 |
| Children Supplemental |     93 |    93 |       0 |

The notebook therefore used **1000 matched TUFTS image-mask pairs** in the analyzed run.

### Dataset Characteristics

The EDA reported:

* Total matched images: **1000**
* Images containing at least one lesion: **339 (33.9%)**
* Mean foreground pixel ratio: **0.00325**
* Approximate background-to-foreground ratio: **306.7:1**
* Average image size: **840 × 1615**
* Image aspect ratio: approximately **1.92**

This demonstrates the highly imbalanced nature of the segmentation task.

---

## 🧠 Approach

### 1. Exploratory Data Analysis

The project performs dataset-level analysis including:

* Image/mask pairing
* Image dimensions
* Aspect ratios
* Foreground pixel ratios
* Lesion presence
* Number of connected lesion components
* Per-source dataset statistics

The image-mask matching logic is designed to handle:

* Different file extensions
* Case differences
* Nested directories
* Prefix/suffix differences
* Inconsistent naming conventions

---

### 2. Image Preprocessing

Images are resized to:

```text
512 × 512
```

The project uses **Albumentations** for image transformations and `ToTensorV2` for PyTorch tensor conversion.

Image normalization uses ImageNet-style statistics:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

### 3. Deep Learning Model

The implementation is based on:

* **PyTorch**
* **Segmentation Models PyTorch (SMP)**

The architecture/model configuration is defined through SMP and is designed for binary segmentation of lesion regions.

---

## ⚙️ Training Configuration

The current configuration includes:

| Parameter                  |     Value |
| -------------------------- | --------: |
| Batch Size                 |        16 |
| Epochs                     |        60 |
| Learning Rate              |      3e-4 |
| Weight Decay               |      1e-4 |
| Input Size                 | 512 × 512 |
| Random Seed                |        42 |
| Early Stopping Patience    |        12 |
| Morphological Kernel       |         3 |
| Minimum Blob Size          |         6 |
| Positive Weight Cap        |        25 |
| Positive Sample Multiplier |         3 |

The training pipeline also performs threshold optimization using thresholds ranging from:

```text
0.10 → 0.85
```

---

## 🎯 Handling Class Imbalance

A major challenge in this dataset is that lesion pixels represent only a very small portion of each panoramic X-ray.

The EDA found an approximate:

```text
Background : Lesion
306.7 : 1
```

To address this, the training configuration incorporates:

* Positive-class weighting
* Positive sample multiplication
* Tversky loss parameters
* Focal loss parameters
* BCE contribution
* Prediction threshold optimization
* Morphological post-processing

Configured parameters include:

```text
Tversky α = 0.35
Tversky β = 0.65
Focal γ   = 0.75
BCE weight = 0.5
```

---

## 🔬 Prediction Pipeline

After model inference, predicted probabilities are converted into binary masks using an optimized threshold.

The project supports morphological post-processing to remove small noisy regions and improve segmentation quality.

Prediction visualization includes:

1. Original OPG X-ray
2. Ground-truth segmentation mask
3. Raw model probability map
4. Prediction overlay

---

## 📊 Visualization

The prediction visualization distinguishes:

* 🟢 **True Positive (TP)** — correctly detected lesion
* 🔴 **False Positive (FP)** — incorrectly detected region
* 🟡 **False Negative (FN)** — missed lesion region

Example output files generated by the notebook:

```text
prediction_sample_1.png
prediction_sample_2.png
prediction_sample_3.png
```

EDA results are saved as:

```text
eda_stats.csv
eda_plots.png
```

---

## 🛠️ Tech Stack

### Programming

* Python 3

### Deep Learning

* PyTorch
* Segmentation Models PyTorch

### Computer Vision

* OpenCV
* NumPy
* Albumentations

### Data Processing

* Pandas
* Scikit-learn

### Visualization

* Matplotlib

### Environment

* Kaggle Notebook
* NVIDIA Tesla T4 environment

---

## 📁 Project Structure

```text
dental-xray-lesion-segmentation/
│
├── d-x-ray.ipynb
├── eda_stats.csv
├── eda_plots.png
│
├── prediction_sample_1.png
├── prediction_sample_2.png
├── prediction_sample_3.png
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd dental-xray-lesion-segmentation
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy pandas matplotlib
pip install scikit-learn
pip install albumentations
pip install segmentation-models-pytorch
```

### 3. Open the Notebook

```bash
jupyter notebook d-x-ray.ipynb
```

Alternatively, the notebook can be executed directly on **Kaggle** after adding the required datasets.

---

## ▶️ Workflow

```text
Dental X-Ray Dataset
        │
        ▼
Image / Mask Pairing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Image & Mask Preprocessing
        │
        ▼
512 × 512 Input
        │
        ▼
Segmentation Model
        │
        ▼
Pixel-wise Probability Map
        │
        ▼
Threshold Optimization
        │
        ▼
Morphological Post-processing
        │
        ▼
Final Lesion Segmentation
        │
        ▼
Visualization & Evaluation
```

---

## 🔍 Key Challenges

### Severe Class Imbalance

Lesion regions occupy a very small percentage of panoramic X-ray images, creating a strong imbalance between background and lesion pixels.

### Inconsistent Dataset Naming

Different datasets use different naming conventions for images and masks. The project therefore implements normalization and matching logic to improve image-mask pairing.

### Small Lesion Regions

Small lesions can easily be lost during resizing or incorrectly classified as background, requiring careful preprocessing, loss design, threshold selection, and post-processing.

---

## 📈 Current EDA Findings

The analyzed dataset contains:

```text
1000 matched X-ray/mask pairs
339 images with lesions
661 images without lesions
306.7 : 1 background-to-lesion pixel ratio
```

The lesion-count distribution ranges from zero lesions to images containing multiple connected lesion regions, with up to **11 detected connected components** in the analyzed masks.

---

## 🔮 Future Improvements

* Integrate the additional children's datasets after resolving image-mask naming mismatches
* Compare multiple segmentation architectures such as U-Net, U-Net++, FPN, and DeepLabV3+
* Perform systematic hyperparameter optimization
* Add Dice, IoU, precision, recall, and F1 evaluation
* Experiment with stronger augmentation strategies
* Improve small-lesion detection
* Perform cross-dataset validation
* Add model checkpointing and experiment tracking
* Deploy the trained model through a REST API or web interface

---

## 👨‍💻 Author

**Musudi Shubhankar Rao**

B.Tech — Computer Science & Engineering

---

## 📜 License

This project is intended for educational and research purposes. Please review the original dataset licenses and terms before using the datasets commercially.
