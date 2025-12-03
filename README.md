# ToN-IoT Cyber Attack Detection

This repository contains a collection of Jupyter notebooks implementing multiple machine learning models for anomaly detection and attack classification on the ToN-IoT dataset. The focus is hands-on exploration of classical ML and neural methods, model evaluation, and practical workflows for cybersecurity analytics.

**Key goals**
- Detect anomalies in IoT network/telemetry data
- Classify attack types vs normal traffic
- Compare different algorithms and feature selection strategies

**What’s included**
- `Support Vector Machines.ipynb`: SVM classifiers (linear/RBF), scaling, hyperparameter tuning
- `Decision Tree & Naive Bayes.ipynb`: Tree-based models and probabilistic baselines
- `Ensemble Learning.ipynb`: Random Forest, Gradient Boosting, Bagging, Voting/Stacking
- `Mutli Layer Perceptron.ipynb`: MLP neural network for tabular intrusion detection
- `Feature Selection.ipynb`: Filter/wrapper methods, importance ranking, dimensionality reduction
- `Clustering.ipynb`: Unsupervised anomaly detection with K-Means/DBSCAN
- `Class Imbalance Methods.ipynb`: Handling skewed labels via resampling and class-weighting


## Dataset
- **ToN-IoT**: A comprehensive dataset for IoT network intrusion and cyber attack research.
- Official page: https://tronxy.github.io/ton-iot/ (or search "ToN-IoT dataset" by the University of New South Wales Canberra Cyber).
- Typical modalities: network traffic, telemetry, and logs. This repo focuses on tabular subsets prepared for supervised/unsupervised learning.

Note: Due to dataset size/licensing, raw files are not versioned here. Follow instructions below to download and prepare your local copy.


## Quick Start
1) Create a Python environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

2) Install core dependencies (adjust as needed for your notebooks)

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn jupyter
```

3) Launch Jupyter and open any notebook

```powershell
jupyter notebook
```

4) Prepare the dataset
- Download ToN-IoT tabular CSVs from the official source
- Place them under a local `data/` folder
- Update notebook paths and any preprocessing cells to match your local file locations


## Workflow Overview
- Data loading and cleaning: handle missing values, parse timestamps, normalize numerics
- Feature engineering: one-hot encoding, scaling, PCA or selection by importance
- Model training: fit multiple algorithms across notebooks
- Evaluation: accuracy, precision/recall, F1, ROC-AUC; confusion matrices
- Class imbalance: try `class_weight`, SMOTE/ADASYN, undersampling, threshold tuning
- Unsupervised: clustering and distance-based outlier scoring


## Notebooks Summary
- `Support Vector Machines.ipynb`: Grid search on `C`, `gamma`; standardized features; decision boundaries; ROC curves
- `Decision Tree & Naive Bayes.ipynb`: Interpretable trees, pruning; Gaussian/Multinomial NB baselines for fast inference
- `Ensemble Learning.ipynb`: Robust bagging/boosting; feature importance; calibration; stacking meta-models
- `Mutli Layer Perceptron.ipynb`: Hidden layers, activations, regularization; early stopping; learning curves
- `Feature Selection.ipynb`: SelectKBest, mutual information, RFE; compare performance vs full feature set
- `Clustering.ipynb`: K-Means silhouette/Davies–Bouldin; DBSCAN for outlier detection; mapping clusters to labels
- `Class Imbalance Methods.ipynb`: SMOTE/Borderline-SMOTE, Tomek links, class-weighting; metric tracking per class


## Results & Evaluation
Results vary by preprocessing, feature set, and imbalance handling. Typical observations:
- Ensembles (Random Forest/Gradient Boosting) are strong baselines
- SVMs perform well with careful scaling and `C/gamma` tuning
- MLPs benefit from balanced batches and regularization
- Feature selection can reduce overfitting and improve latency
- Handling class imbalance significantly improves minority-class F1

Each notebook includes metrics like precision, recall, F1-score, confusion matrices, and sometimes ROC-AUC. For reproducible reporting, consider exporting results to CSV and aggregating across runs.


## Repository Structure
- `*.ipynb`: Individual experiments per algorithm/topic
- `data/`: Place local ToN-IoT CSV files here (ignored in VCS)
- `README.md`: This documentation


## Reproducibility Tips
- Fix random seeds where possible (e.g., `np.random.seed`, model `random_state`)
- Use stratified splits for classification tasks
- Keep a consistent preprocessing pipeline (scikit-learn `Pipeline`)
- Log model configs and metrics for comparisons


## References
- ToN-IoT Dataset (UNSW Canberra Cyber) — search for the official repository/publications
- scikit-learn documentation: https://scikit-learn.org/stable/
- imbalanced-learn: https://imbalanced-learn.org/stable/


## Next Steps
- Add `requirements.txt` and a data preparation script
- Integrate `Pipeline`/`GridSearchCV` utilities for consistent runs
- Expand to time-series models for sequential telemetry
- Add experiment tracking (e.g., MLflow) for results comparison
