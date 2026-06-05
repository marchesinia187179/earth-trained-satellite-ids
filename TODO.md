# Project TODO List

## Completed Tasks
- [x] **Project Infrastructure**: Established modular directory structure and basic utility functions for file and directory management.
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
    - [x] Evaluation metrics calculation (F1-score, Precision, Recall).
    - [x] Model persistence (saving to `.joblib`) and metadata logging to CSV.
- [x] **CLI Orchestration**: Created a main entry point to handle user interaction for preprocessing and model building loops.

## Future Tasks
- [ ] **Random Forest Evaluation**: Implement the testing phase for the Random Forest model specifically using STIN data (terrestrial vs. satellite cross-evaluation).
- [ ] **Isolation Forest Implementation**: 
    - [ ] Implement the `isolation_forest` model logic in `models.py`.
    - [ ] Develop the relative testing and evaluation suite for anomaly detection.
- [ ] **Hybrid System Integration**: Research and implement the feasibility study of combining both models into a Hybrid IDS.