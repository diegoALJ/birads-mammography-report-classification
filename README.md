# BI-RADS Mammography Report Classification

This repository contains my solution and experimentation workflow for the **SPR 2026 Mammography Report Classification** Kaggle competition, focused on predicting the **BI-RADS category (0–6)** from the **text of mammography reports**, using only the **indication** and **findings** sections of each report.

The project explores **transformer-based NLP approaches** for clinical text classification, primarily using the pretrained models **`pucpr/biobertpt-clin`** and **BERTimbau**, with the goal of inferring radiological assessment from descriptive mammography findings alone.

**Best competition score:** **0.83**  
**Kaggle competition:** (https://www.kaggle.com/competitions/spr-2026-mammography-report-classification)

---

## Project Overview

The competition was organized by the **Radiology Society of São Paulo (SPR - Sociedade Paulista de Radiologia)** as part of its Artificial Intelligence Challenge for Breast Cancer. The objective is to develop a model that accurately predicts the BI-RADS category from mammography report text, excluding the impression/conclusion section where the target label is explicitly stated.

This task is clinically relevant because it simulates a decision-support setting in which a model must infer the final assessment from the descriptive content of the report. Potential applications include:

- improving consistency across radiology reports,
- supporting automated quality control,
- providing educational feedback for trainees,
- and enabling large-scale research in breast imaging NLP.

The competition metric is **macro F1-score** over the seven BI-RADS classes.

---

## About BI-RADS

**BI-RADS** (Breast Imaging Reporting and Data System) is a standardized framework developed by the **American College of Radiology (ACR)** for breast imaging reporting. In this challenge, the target is a **7-class classification problem**:

- **0** – Incomplete, needs additional imaging
- **1** – Negative
- **2** – Benign finding
- **3** – Probably benign
- **4** – Suspicious abnormality
- **5** – Highly suggestive of malignancy
- **6** – Known biopsy-proven malignancy

The target label was extracted from the original report’s **impression/conclusion** section, which was removed from the input text provided to participants.

---

## Dataset

The dataset consists of **de-identified mammography radiology reports** collected from multiple Brazilian institutions. Each report typically contains three sections:

1. **Indication / Reason for Exam**
2. **Findings**
3. **Impression / Conclusion**

For this competition, participants are given only the **indication** and **findings** sections, while the **impression/conclusion** is excluded because it contains the ground-truth BI-RADS label.

### Files
- **`train.csv`**  
  Contains labeled reports with:
  - `id`
  - `report`
  - `target`

- **`test.csv`**  
  Contains unlabeled reports with:
  - `id`
  - `report`

- **`submission.csv`**  
  Submission file with:
  - `id`
  - `target`

---

## Important Data Availability Note

The competition data is subject to **strict privacy and usage restrictions**. According to the challenge rules, participants agree **not to copy, redistribute, or attempt to re-identify any part of the dataset**. For that reason:

- **the original competition dataset is not included in this repository**,
- **no raw or processed competition data is shared here**,
- and this repository is intended to provide the **code, project structure, and reproducible workflow only**, in accordance with the competition’s data access constraints.

To reproduce the experiments, you must obtain access to the dataset directly through the official Kaggle competition page and comply with all competition rules.

---

---

## Modeling Approach

This project focused on **transformer-based clinical NLP classification**, primarily using:

- **`pucpr/biobertpt-clin`**
- **BERTimbau**

Among the tested backbones, **`pucpr/biobertpt-clin` performed slightly better overall**, although the gap versus BERTimbau was not large.

### Training setup

- **5-fold cross-validation**
- typically **8 to 12 epochs**
- **early stopping patience of 3 epochs**
- model selection based on **macro F1**

The goal was to evaluate how well Portuguese biomedical and general-domain transformer models could infer BI-RADS classes from the descriptive sections of mammography reports alone.

---

## Preprocessing

Preprocessing was intentionally minimal.

Based on exploratory data analysis, the reports were already well-structured and consistently organized, so no heavy text normalization pipeline was necessary. The main preprocessing step focused on **leakage cleaning**, ensuring that no direct target information from the impression/conclusion section remained in the modeling input.

This design choice helped preserve the original clinical language while minimizing the risk of artificial performance inflation.

---

## Key Results

- **Best competition score:** **0.83**
- **Validation strategy:** 5-fold cross-validation
- **Best-performing backbone:** `pucpr/biobertpt-clin`
- **Alternative backbone explored:** BERTimbau
- **Core metric:** macro F1

This project showed that pretrained transformer models can perform strongly on radiology report classification with relatively limited preprocessing, provided that leakage is handled carefully and validation is done properly.

---
## Repository Structure

This repository is being organized progressively into a more modular format. Initially, the experimentation workflow is centered around **two notebooks**:

- **`01_eda.ipynb`** – exploratory data analysis
- **`02_modeling.ipynb`** – feature engineering, modeling, validation, ensembling, and inference


```
birads-mammography-report-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── modeling.py
│   ├── inference.py
│   └── utils.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── submissions/
├── images/
│   └── README_assets/
└── data/
```

---

## Reproducibility

Because the dataset cannot be redistributed, this repository is designed to be reproducible once the user has access to the official competition data.

---

## General workflow:

Join the Kaggle competition and accept its rules.
Download the competition data from Kaggle.
Place the files locally in the expected directory structure.
Run the notebooks and/or modularized scripts.
Generate predictions and submission files.

More detailed setup instructions can be added once the final project structure is completed.

---


---

## Organizers

This competition was organized by:

Sociedade Paulista de Radiologia
Douglas Racy
Gustavo Corradi
Eduardo Farina
Vanderlei Silva
Aline Sessino
Giuliano Giovanetti
Almir Bittencourt
Felipe Kitamura
Lilian Mallagoli
Sheila Costa
Donating Institutions
AC Camargo
Almir Bittencourt
Soraia Damião
Hapvida
Alexandre Bialowas
Eduardo Caminha
Unifesp
Eduardo Farina
Nitamar Abdala
Citation

If you use or reference the competition, please cite:

Eduardo Farina and Felipe Kitamura, MD, PhD.
SPR 2026 Mammography Report Classification.
Kaggle, 2026.
(https://www.kaggle.com/competitions/spr-2026-mammography-report-classification)

---

## Disclaimer

This repository is an independent portfolio project built around a public Kaggle competition. It is intended for educational, research, and reproducibility purposes only. All dataset rights, competition rules, and privacy restrictions remain with the official organizers and hosting platform.
    ├── sample/
    └── metadata/
