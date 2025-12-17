# Loan Eligibility Prediction - Advanced ML Ensemble

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📊 Project Overview

This project implements an advanced machine learning ensemble system for predicting loan eligibility based on applicant financial and demographic characteristics. The solution leverages multiple ML algorithms with sophisticated feature engineering, cross-validation, and model blending strategies to achieve robust predictions.

### Key Highlights
- **Multi-model ensemble** combining Random Forest, Logistic Regression, Gradient Boosting, XGBoost, LightGBM, and Neural Networks
- **Advanced feature engineering** with 20+ derived features capturing complex interactions
- **Stratified K-Fold cross-validation** with bagging for robust model evaluation
- **Intelligent blending** using score-weighted ensemble strategies
- **Class imbalance handling** through both down-sampling and up-sampling techniques

---

## 📁 Dataset Description

### Files
- `train.csv` - Training dataset with 58,646 samples
- `test.csv` - Test dataset with 39,098 samples for predictions
- `Loan_Eligibility_Prediction.py` - Complete model pipeline implementation

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numerical | Age of the loan applicant |
| `person_income` | Numerical | Annual income of the applicant |
| `person_home_ownership` | Categorical | Home ownership status (RENT, OWN, MORTGAGE, OTHER) |
| `person_emp_length` | Numerical | Employment length in years |
| `loan_intent` | Categorical | Purpose of the loan (EDUCATION, MEDICAL, VENTURE, PERSONAL, etc.) |
| `loan_grade` | Categorical | Loan grade assigned (A through G) |
| `loan_amnt` | Numerical | Loan amount requested |
| `loan_int_rate` | Numerical | Interest rate on the loan |
| `loan_percent_income` | Numerical | Loan amount as percentage of income |
| `cb_person_default_on_file` | Categorical | Historical default indicator (Y/N) |
| `cb_person_cred_hist_length` | Numerical | Length of credit history in years |
| **`loan_status`** | **Target** | **Binary outcome: 0 (approved) or 1 (default)** |

---

## 🔧 Feature Engineering

The pipeline creates 20+ engineered features to capture complex relationships:

### Interaction Features
- `age_income_interaction` - Multiplicative interaction between age and income
- `loan_to_emp_length_ratio` - Loan amount relative to employment tenure
- `dti_ratio` - Debt-to-income ratio (monthly debt / monthly income)
- `risk_interaction` - Interaction between default history and loan grade

### Ratio Features
- `person_income_to_age` - Income growth rate over lifetime
- `loan_amnt_to_income_ratio` - Loan burden relative to income
- `emp_length_to_age_ratio` - Career stability indicator
- `interest_to_income_ratio` - Interest payment capacity
- `income_credit_ratio` - Income relative to credit history length

### Polynomial Features
- `person_age_squared` - Non-linear age effects
- `person_income_log` - Log-transformed income for skewed distribution
- `loan_amnt_log` - Log-transformed loan amount

### Binned Features
- `age_bin` - Age discretized into 5 categories
- `income_bin` - Income discretized into 5 categories
- `loan_amnt_bin` - Loan amount discretized into 5 categories

---

## 🤖 Model Architecture

### Base Models

1. **Random Forest Classifier**
   - 200 estimators, max depth 15
   - Class-balanced weighting
   - Parallel processing enabled

2. **Logistic Regression**
   - L2 regularization (C=0.1)
   - Class-balanced weighting
   - Liblinear solver for stability

3. **Multi-Layer Perceptron (Neural Network)**
   - Architecture: 128 → 64 → 32 neurons
   - ReLU activation, Adam optimizer
   - Early stopping with validation

4. **Gradient Boosting (sklearn)**
   - 200 estimators, max depth 6
   - Learning rate 0.1, subsample 0.8

5. **XGBoost** (if available)
   - 300 estimators, max depth 6
   - Learning rate 0.05, advanced regularization

6. **LightGBM** (if available)
   - 300 estimators, optimized for speed
   - Class-balanced weighting

### Ensemble Strategy

The final prediction uses **score-weighted blending**:
```python
weight_i = max(score_i - min_score + 0.01, 0.01) / Σ(weights)
final_prediction = Σ(weight_i × prediction_i)
```

This approach assigns higher weights to better-performing models while preventing any single model from dominating.

---

## 🔄 Cross-Validation Pipeline

### Stratified K-Fold with Bagging
- **5-fold stratified cross-validation** preserves class distribution
- **2 bags per fold** with alternating sampling strategies:
  - Bag 1: Down-sampling majority class
  - Bag 2: Up-sampling minority class
- **10 total model fits** per base model (5 folds × 2 bags)

### Out-of-Fold (OOF) Predictions
- Each validation fold receives predictions from models trained on remaining folds
- OOF predictions enable unbiased performance estimation
- Used for computing ROC-AUC score on full training set

---

## 📈 Performance Metrics

- **Primary Metric**: ROC-AUC (Area Under Receiver Operating Characteristic Curve)
- **Evaluation**: Out-of-fold predictions on training data
- **Calibration**: Predictions clipped to [0.001, 0.999] range

### Achieved Performance

**Final Ensemble ROC-AUC: 0.9495632**

Individual Model Performance:
| Model | ROC-AUC Score | Ensemble Weight |
|-------|---------------|-----------------|
| Gradient Boosting (GB) | 0.9526750 | 0.4822 (48.2%) |
| Random Forest (RF) | 0.9353670 | 0.3245 (32.5%) |
| Multi-Layer Perceptron (MLP) | 0.9109790 | 0.1021 (10.2%) |
| Logistic Regression (LOGREG) | 0.9097750 | 0.0912 (9.1%) |

The ensemble achieves **94.96% ROC-AUC** through intelligent score-weighted blending, with Gradient Boosting contributing nearly half the final prediction due to its superior performance.

---

## 🚀 Usage

### Requirements
```bash
pip install numpy pandas scikit-learn scipy
pip install xgboost lightgbm  # Optional but recommended
```

### Running the Pipeline
```bash
python Loan_Eligibility_Prediction.py
```

### Output
The script generates:
1. **Console output** with detailed fold-by-fold performance
2. **Feature importance analysis** highlighting top predictive features
3. **Submission file** named `submission_enhanced_{score}_{timestamp}.csv`

---

## 📊 Example Output

![Terminal Output](https://github.com/AmrishS2004/Loan_Eligibility_Prediction/blob/main/Output.jpg)
*Actual model training output showing ensemble performance*

```
Submission saved as: submission_enhanced_0.949563_20251102_152755.csv

Model Performance Summary:
RF: Score = 0.935367, Weight = 0.3245
LOGREG: Score = 0.909775, Weight = 0.0912
GB: Score = 0.952675, Weight = 0.4822
MLP: Score = 0.910979, Weight = 0.1021

Final Score: 0.9495632
```

### Performance Analysis
- **Gradient Boosting** emerged as the strongest single model (95.27% AUC)
- **Random Forest** provided stable secondary performance (93.54% AUC)
- Neural network and logistic regression contributed complementary predictions
- Score-weighted blending allocated nearly 50% weight to GB due to superior performance
- Final ensemble achieved **94.96% ROC-AUC** on out-of-fold predictions

---

## 🔍 Key Implementation Details

### Class Imbalance Handling
The dataset exhibits class imbalance (more approved loans than defaults). The pipeline addresses this through:
- Alternating down-sampling and up-sampling during bagging
- Class-balanced weights in tree-based models
- Stratified sampling in cross-validation

### Preprocessing Pipeline
- **Numerical features**: Median imputation + Standard scaling
- **Categorical features**: Most-frequent imputation + One-hot encoding
- **Missing values**: Handled gracefully through imputation strategies

### Computational Efficiency
- Parallel processing for Random Forest (`n_jobs=-1`)
- Garbage collection between folds to manage memory
- Optional models (XGBoost/LightGBM) gracefully skipped if unavailable

---

## 📌 Model Selection Notes

The pipeline automatically detects available libraries:
- **Core models** (always available): Random Forest, Logistic Regression, Gradient Boosting, MLP
- **Optional models** (if installed): XGBoost, LightGBM
- Failed model fits are caught and skipped without breaking the pipeline

---

## 🎯 Future Improvements

- [ ] Hyperparameter tuning using Optuna or GridSearchCV
- [ ] Additional feature engineering based on domain expertise
- [ ] Stacking meta-learner for second-level ensemble
- [ ] SHAP values for model interpretability
- [ ] Handling outliers with robust scaling techniques

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Amrish Sasikumar**
- Email: amrish.s2004p@gmail.com
- LinkedIn: [linkedin.com/in/amrish-sasikumar](https://www.linkedin.com/in/amrish-s-2aa191220/)

---

## 🙏 Acknowledgments

- Dataset source: [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s4e10)
- Inspired by best practices in Kaggle competitions and production ML systems
- Built with scikit-learn, XGBoost, and LightGBM ecosystems

---

*Last Updated: December 2024*
