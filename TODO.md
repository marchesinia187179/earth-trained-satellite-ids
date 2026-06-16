# Project TODO List

## Completed Tasks
- [x] **Project Infrastructure**: Established modular directory structure and centralized path management (`paths.py`).
- [x] **Centralized Utilities**: Implemented project-wide input validation and user interaction logic (`input_utils.py`).
- [x] **Data Preprocessing**:
    - [x] Implementation of dataset alignment for NB15 and STIN (ter20, sat20) datasets.
    - [x] Feature normalization using Min-Max scaling.
    - [x] Minority class removal for NB15 and class merging for STIN.
- [x] **File Preprocessing**:
    - [x] Automated splitting of datasets by attack category.
    - [x] Implementation of balancing logic (Normal/Attack ratio) for training sets.
    - [x] Functionality to merge specific attack categories for comprehensive training.
- [x] **Random Forest Model**:
    - [x] Implementation of the training pipeline with optional hyperparameter tuning (RandomizedSearchCV).
    - [x] Model persistence (saving to `.joblib`) and detailed metadata logging (`models_info.csv`) including TP, TN, FP, FN.
- [x] **Isolation Forest Implementation**: 
    - [x] Implemented the `isolation_forest` model logic in `models.py` including proper training on normal data.
    - [x] Integrated reporting and evaluation for anomaly detection with specific label mapping (-1/1 to 1/0).
- [x] **Classification & Evaluation**:
    - [x] Implemented evaluation loop for testing models on cross-domain datasets (NB15, SAT20, TER20).
    - [x] Standardized reporting with model-specific CSV files and comprehensive metrics (F1, Precision, Recall, AUC-ROC).
    - [x] Implemented metric robustness: handled single-class test sets by setting invalid metrics to 'None' and fixing 2x2 confusion matrix dimensions.

## Future Tasks
- [ ] **Data Visualization**: Develop scripts to visualize performance metrics and domain shift effects from `results.csv`.
- [ ] **Advanced Evaluation**: Implement automated plotting for ROC and Precision-Recall curves to better analyze model thresholds.
