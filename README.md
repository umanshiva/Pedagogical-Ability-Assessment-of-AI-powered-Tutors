# Pedagogical Ability Assessment of AI-powered Tutors

This repository contains the implementation for **assessing the pedagogical effectiveness of AI-powered tutors** on a custom dataset of tutor–student dialogues.  
We fine-tune transformer-based models (BERT / RoBERTa) with advanced loss functions and explore multiple training strategies to classify tutor responses across four pedagogical dimensions:

- Mistake Identification  
- Mistake Location  
- Pedagogical Guidance  
- Actionability

---

## 📌 Project Overview
This work evaluates the ability of AI tutors to provide effective pedagogical feedback.  
Key highlights:
- Dataset: 300 tutor–student dialogues (MathDial + Bridge).  
- Evaluation: **Exact** (3 classes) and **Lenient** (2 classes) settings.  
- Models: BERT-base and RoBERTa-base.  
- Loss functions: Cross-Entropy, Label Smoothing, Focal Loss, Weighted CE.  
- Data augmentation via SMOTE to handle class imbalance.

---
## 📂 Repository Structure
```
pedagogical-assessment/
│
├── README.md                          # Project overview, usage, methods
├── requirements.txt                   # Python dependencies
├── preprocessing_pipeline.py          # Preprocess & balance the dataset (.json)
│
├── common_backbone_*.py               # Shared backbone + task-specific heads (all-in-one)
├── individual_*.py                    # Independent backbone for each task (per-loss/per-setting)
│
├── notebooks/                         # Jupyter notebooks for experiments (run independently)
│   ├── bert_common_backbone.ipynb
│   ├── roberta_common_backbone.ipynb
│   ├── individual_mistake_identification.ipynb
│   └── ...
│
└── data/                              # Dataset (.json) & processed splits
```

---

## 🧭 Usage Guide

### Preprocess the Dataset
Use `preprocessing_pipeline.py` to clean and prepare data.

```bash
python preprocessing_pipeline.py
```

To enable **balanced sampling**, call:
```python
from preprocessing_pipeline import preprocess_data
preprocess_data(mode='balanced', task='mistake_identification')
```

---
### Training Strategies
We experimented with two approaches:

- **Common Backbone with Task-Specific Heads**  
  - Files starting with `common_...`  
  - A shared encoder (BERT / RoBERTa) for all tasks, with independent classification heads.

- **Individual Backbone per Task**  
  - Files starting with `individual_...`  
  - Separate encoder + classifier for each task.

> Example:  
> `individual_bert_exact_all_losses_mistake_identification.py`  
> → Implements BERT in **exact** mode for Mistake Identification, testing multiple losses.

---

### Loss Functions & Settings
- **Cross-Entropy (CE)** – baseline.  
- **Label Smoothing (LS)** – regularization.  
- **Focal Loss (FL)** – focus on hard examples.  
- **Weighted Cross-Entropy** – for class imbalance.

File names indicate the loss/setting used.

---

### Running Experiments
Each notebook in `notebooks/` can be run independently:

```bash
jupyter notebook notebooks/<notebook_name>.ipynb
```

Trained models will output Accuracy and Macro F1 for both Exact and Lenient evaluation settings.

---

## 📊 Methods & Findings
- **Models:** BERT-base, RoBERTa-base.  
- **Strategies:** common vs. individual backbones.  
- **Losses:** CE, Label Smoothing, Focal Loss, Weighted CE.  
- **Augmentation:** SMOTE improved F1-scores for underrepresented classes.  

> BERT with a shared backbone + Cross-Entropy achieved the best overall accuracy.  
> Balanced augmentation further boosted macro F1 in lenient evaluation.

---

## 🧪 Results Summary
- **Common Backbone:** Best with BERT + Cross-Entropy (Accuracy up to 0.82 in exact setting).  
- **Individual Models:** Helped certain tasks (e.g., Mistake Identification).  
- **Lenient Evaluation:** Consistently higher F1-scores, reflecting better handling of partial correctness.  
- **Data Augmentation:** SMOTE yielded notable gains across tasks.

---

## 👥 Authors
- Umanshiva Ladva  
- Rajiv Chaudhary  
- Siddhesh Gholap  

---

## 📝 Notes
- Keep `.json` dataset files in `data/`.  
- To add a new loss or setting, create a corresponding `individual_*` script with a clear name.  
- Refer to the original report for detailed analysis and tables.
