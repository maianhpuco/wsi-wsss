# wsi-wsss

install taming model 
python 3.10 
conda env update --file src/includes/taming-transformers/environment.yaml  --name wsi-safe 
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers 
pip install timm  

### Patch-level dataset:
Datasets will be unpacked here.

WSSS-Tissue/

|_ datasets
|     |_ BCSS-WSSS/
|         |_ train/
|         |_ val/
|             |_ img/
|             |_ mask/
|         |_ test/
|             |_ img/
|             |_ mask/
|     |_ LUAD-HistoSeg/
|         |_ train/
|         |_ val/
|             |_ img/
|             |_ mask/
|         |_ test/
|             |_ img/
|             |_ mask/ 


### Slide-level dataset:




reference repo: 
[9]WSSS-Tissue:  https://github.com/ChuHan89/WSSS-Tissue 

# 📄 Dataset Summaries

## BCSS-WSSS Dataset

**BCSS-WSSS** is a weakly-supervised tissue semantic segmentation dataset based on the Breast Cancer Semantic Segmentation (BCSS) dataset.

- **Task**: Patch-level classification and pixel-level segmentation.
- **Tissue Categories**:
  - Tumor (TUM)
  - Stroma (STR)
  - Lymphocytic infiltrate (LYM)
  - Necrosis (NEC)

### 📂 Folder Structure

| Split | Path | Notes |
|:---|:---|:---|
| **Train** | `_train/annotations/` | - Patch-level labels embedded in filenames<br>- Total patches: **23,422** |
| **Validation** | `_val/` (`_img/` and `_mask/`) | - Pixel-level segmentation masks<br>- Total patches: **3,418** |
| **Test** | `_test/` (`_img/` and `_mask/`) | - Pixel-level segmentation masks<br>- Total patches: **4,986** |

Other files:
- `_Readme.txt`
- `_Example_for_using_image_level_label.py`

### 🏷️ Naming Convention (Training Set)

```
Image-name-of-BCSS + '+' + index + '[' + abcd + '].png
```
- Example:  
  `TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+[1101].png`

Where:
- `a`: Tumor (TUM)
- `b`: Stroma (STR)
- `c`: Lymphocyte (LYM)
- `d`: Necrosis (NEC)

Each patch is **224×224** in size.

### 🎨 Palette (Validation/Test Masks)

| Label | Value | RGB Color |
|:---|:---|:---|
| Background | 0 | Black |
| Tumor (TUM) | 1 | `[255, 0, 0]` |
| Stroma (STR) | 2 | `[0, 255, 0]` |
| Lymphocyte (LYM) | 3 | `[0, 0, 255]` |
| Necrosis (NEC) | 4 | `[255, 255, 0]` |

---

## LUAD-HistoSeg Dataset

**LUAD-HistoSeg** is a weakly-supervised tissue semantic segmentation dataset based on lung adenocarcinoma histology slides.

- **Task**: Patch-level classification and pixel-level segmentation.
- **Tissue Categories**:
  - Tumor epithelial (TE)
  - Tumor-associated stroma (TAS)
  - Lymphocyte (LYM)
  - Necrosis (NEC)

### 📂 Folder Structure

| Split | Path | Notes |
|:---|:---|:---|
| **Train** | `_training/` | - Patch-level labels embedded in filenames<br>- Total patches: **16,678** |
| **Validation** | `_val/` (`_img/` and `_mask/`) | - Pixel-level segmentation masks<br>- Total patches: **300** |
| **Test** | `_test/` (`_img/` and `_mask/`) | - Pixel-level segmentation masks<br>- Total patches: **307** |

Other files:
- `_Readme.txt`
- `_Example_for_using_image_level_label.py`

### 🏷️ Naming Convention (Training Set)

```
patient_ID + '_' + x-axis + '-' + y-axis + '[' + a b c d + '].png
```
- Example:  
  `1031280-2300-27920-[1 0 0 1].png`

Where:
- `a`: Tumor epithelial (TE)
- `b`: Necrosis (NEC)
- `c`: Lymphocyte (LYM)
- `d`: Tumor-associated stroma (TAS)

Each patch is **224×224** in size at **10× magnification**.

### 🎨 Palette (Validation/Test Masks)

| Label | Value | RGB Color |
|:---|:---|:---|
| Tumor epithelial (TE) | 0–2 | `[205, 51, 51]` |
| Necrosis (NEC) | 3–5 | `[0, 255, 0]` |
| Lymphocyte (LYM) | 6–8 | `[65, 105, 225]` |
| Tumor-associated stroma (TAS) | 9–12 | `[255, 165, 0]` |
| Background (Exclude) | 12–15 | `[255, 255, 255]` |

---

# 🔥 Quick Comparison

| Aspect | BCSS-WSSS | LUAD-HistoSeg |
|:---|:---|:---|
| Disease | Breast Cancer | Lung Adenocarcinoma |
| Label Extraction | `[abcd]` in filename | `[a b c d]` in filename |
| Training Crop | Sliding window | Random anchor points |
| Magnification | ~20× | 10× |
| Patch Size | 224×224 | 224×224 |
| Categories | TUM, STR, LYM, NEC | TE, TAS, LYM, NEC |

---

# 📌 Notes
- Training sets are **weakly-labeled** (from filenames).
- Validation and Test sets are **pixel-labeled** (ground truth masks).
- Example scripts are provided to handle patch-level labels.

---

# ✅ End of Summary

------
NEW FOLDER STRUCTURE

LUAD-HistoSeg_organized
|   ├─docs
|       └─Readme.txt
|   ├─test
|   |   ├─img
|   |   |   ├─308 .png images
|       └─mask
|       |   ├─307 .png images
|   ├─train
|   |   ├─16678 .png images
|   ├─train_PM
|   |   ├─PM_b4_5
|   |   |   ├─16678 .png images
|   |   ├─PM_b5_2
|   |   |   ├─16678 .png images
|       └─PM_bn7
|       |   ├─16678 .png images
    └─val
    |   ├─img
    |   |   ├─300 .png images
        └─mask
        |   ├─300 .png images 

BCSS-WSSS_organized
|   ├─docs
|       └─Readme.txt
|   ├─test
|   |   ├─img
|   |   |   ├─4986 .png images
|       └─mask
|       |   ├─4986 .png images
|   ├─train
|   |   ├─23422 .png images
|   ├─train_PM
|   |   ├─PM_b4_5
|   |   |   ├─23422 .png images
|   |   ├─PM_b5_2
|   |   |   ├─23422 .png images
|       └─PM_bn7
|       |   ├─23422 .png images
    └─val
    |   ├─img
    |   |   ├─3418 .png images
        └─mask
        |   ├─3418 .png images