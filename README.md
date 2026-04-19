# BI-RADS Mammography Report Classification

This repository contains my solution and experimentation workflow for the **SPR 2026 Mammography Report Classification** Kaggle competition, focused on predicting the **BI-RADS category (0–6)** from the **text of mammography reports**, using only the **indication** and **findings** sections of each report.

The project explores **transformer-based NLP approaches** for clinical text classification, primarily using the pretrained models **`pucpr/biobertpt-clin`** and **BERTimbau**, with the goal of inferring radiological assessment from descriptive mammography findings alone.

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

## Modeling Approach

This project focused on **clinical NLP classification using pretrained transformer models**, primarily:

- **`pucpr/biobertpt-clin`**
- **BERTimbau**

The main objective was to evaluate how well Portuguese biomedical and general-language transformer backbones could infer BI-RADS categories from the descriptive sections of mammography reports.

The overall workflow included:

- exploratory data analysis (EDA),
- text inspection and preprocessing,
- transformer fine-tuning,
- cross-validation,
- macro F1-based model selection,
- and Kaggle submission generation.

---

## Repository Structure

```text
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
    ├── sample/
    └── metadata/
