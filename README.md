# Classification Model for WISDM Smartphone & Smartwatch Activity Dataset

This project develops and evaluates machine learning models for classifying human fitness activities using the **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset**.

---

## Dataset Overview

- Collected from **51 subjects** performing **18 activities** (e.g., walking, jogging, climbing stairs).
- Data sourced from smartphones and smartwatches equipped with **inertial sensors** (accelerometer & gyroscope).
- 4 `.arff` files:
  - Phone Accelerometer
  - Phone Gyroscope
  - Watch Accelerometer
  - Watch Gyroscope  
- Each file contains **91 features**, leading to a **total of 364 features**.

---

## Goal

Build a robust classification model that can **accurately predict human activity** from sensor data.

---

## Models Used

### 1. Random Forest
An ensemble method using multiple decision trees.

** Pros:**
- High accuracy  
- Good feature importance estimates  
- Fast test error estimation  

### 2. Gradient Boosting
Sequentially builds models to fix previous errors.

** Pros:**
- Top-tier predictive power
- Flexible and customizable
- Handles missing data

** Cons:**
- Slow training
- Risk of overfitting without regularization
- Used default hyperparameters for faster testing.

### 3. Decision Tree
A simple and interpretable model.

** Pros:**
- No data scaling required
- Handles missing values well
- Easy to explain

** Cons:**
- Lower accuracy compared to ensemble methods

## Project Pipeline

### 1. Data Preprocessing
- Loaded .arff files and combined all sensor data.
- Converted labels from A–S to integers 1–18.
- Merged features by timestamp.
- Shuffled the dataset.

### 2. Hyperparameter Tuning
Used GridSearchCV from sklearn:

**Random Forest:**
- Best: `n_estimators=1000, max_depth=30, criterion='entropy'` → Accuracy: 0.884
- Chosen (faster): `n_estimators=100, max_depth=20, criterion='entropy'` → Accuracy: 0.882

**Decision Tree:**
- Tuned: `max_depth=20, criterion='entropy'`

**Gradient Boosting:**
- Used default settings due to training time.

### 3. Feature Selection
- Total features: 364
- Feature importance visualized with correlation heatmaps.
- Used:
  - SelectFromModel (with Random Forest, threshold=0.005)
  - PCA to reduce to 50 components

### 4. Model Training
- Train/test split: 70/30
- Evaluation metric: accuracy
- Also tested VotingClassifier combining all three models.

## Evaluation Results

### ➤ Using All 364 Features
| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.8827   |
| Decision Tree     | 0.7656   |
| Gradient Boosting | 0.8803   |

### ➤ Using 55 Features (SelectFromModel)
| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.8314   |
| Decision Tree     | 0.7325   |
| Gradient Boosting | 0.8010   |
| Voting Classifier | 0.8290   |

### ➤ Using 55 Features (PCA)
| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | 0.8954 ✅ |
| Decision Tree     | 0.6748   |
| Gradient Boosting | 0.8142   |
| Voting Classifier | 0.8606   |

## Final Verdict
🎯 **PCA + Random Forest** achieved the highest accuracy of 0.8954 and is selected as the final model due to its superior performance and efficiency in high-dimensional space.

## Project Structure
```bash
├── data/
│   └── *.arff               # Raw sensor data
├── models/
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   └── decision_tree.py
├── feature_selection/
│   ├── pca.py
│   └── select_from_model.py
├── train.py                 # Main training pipeline
├── evaluate.py              # Evaluation logic
├── README.md

