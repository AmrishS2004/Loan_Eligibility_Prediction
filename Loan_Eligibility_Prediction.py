# Import Libraries
import numpy as np
import pandas as pd
import datetime
import gc
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# Try to import optional packages
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available, skipping...")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("LightGBM not available, skipping...")
    LGB_AVAILABLE = False

# ---------------------------
# Load Data
# ---------------------------
train_data = pd.read_csv("/Users/amrish/Downloads/playground-series-s4e10/train.csv")
test_data = pd.read_csv("/Users/amrish/Downloads/playground-series-s4e10/test.csv")

# ---------------------------
# Enhanced Preprocessing Function
# ---------------------------
def preprocess_enhanced(X):
    X = X.copy()
    
    # Original features
    X['age_income_interaction'] = X['person_age'] * X['person_income']
    X['loan_to_emp_length_ratio'] = X['loan_amnt'] / (X['person_emp_length'] + 1)
    monthly_income = X['person_income'] / 12
    X['monthly_debt'] = X['loan_amnt'] * (1 + X['loan_int_rate']/100) / 12  # Fixed: rate is percentage
    X['dti_ratio'] = X['monthly_debt'] / (monthly_income + 1)
    
    # New engineered features
    X['person_income_to_age'] = X['person_income'] / (X['person_age'] + 1)
    X['loan_amnt_to_income_ratio'] = X['loan_amnt'] / (X['person_income'] + 1)
    X['emp_length_to_age_ratio'] = X['person_emp_length'] / (X['person_age'] + 1)
    X['interest_to_income_ratio'] = X['loan_int_rate'] / (X['person_income'] + 1)
    X['loan_per_year_employed'] = X['loan_amnt'] / (X['person_emp_length'] + 1)
    X['income_credit_ratio'] = X['person_income'] / (X['cb_person_cred_hist_length'] + 1)
    
    # Create risk flag - fixed to handle missing values
    risk_condition = ((X['cb_person_default_on_file'] == 'Y') & 
                     (X['loan_grade'].isin(['C', 'D', 'E', 'F'])))
    X['risk_interaction'] = risk_condition.astype(int) * X['loan_int_rate']
    
    # Polynomial features
    X['person_age_squared'] = X['person_age'] ** 2
    X['person_income_log'] = np.log1p(np.abs(X['person_income']))  # Handle negative values
    X['loan_amnt_log'] = np.log1p(X['loan_amnt'])
    
    # Binning features
    X['age_bin'] = pd.cut(X['person_age'], bins=5, labels=False)
    X['income_bin'] = pd.cut(X['person_income'], bins=5, labels=False)
    X['loan_amnt_bin'] = pd.cut(X['loan_amnt'], bins=5, labels=False)
    
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    numerical_cols = [col for col in X.columns if col not in categorical_cols + ['id', 'loan_status'] and X[col].dtype in ['int64', 'float64']]

    # Transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_transformed = preprocessor.fit_transform(X)
    cat_col_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_col_names = numerical_cols + list(cat_col_names)

    return pd.DataFrame(X_transformed, columns=all_col_names), preprocessor

# ---------------------------
# Enhanced Down-sampling
# ---------------------------
def balanced_sampling(X, y, method='downsample', random_state=42):
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if method == 'downsample':
        majority_class = X[y == 0]
        minority_class = X[y == 1]
        
        if len(majority_class) > len(minority_class):
            majority_sample = majority_class.sample(len(minority_class), random_state=random_state)
            X_balanced = pd.concat([majority_sample, minority_class], axis=0)
            y_balanced = pd.concat([y.iloc[majority_sample.index], y[y == 1]], axis=0)
        else:
            X_balanced = X.copy()
            y_balanced = y.copy()
            
    elif method == 'upsample':
        minority_class = X[y == 1]
        if len(minority_class) > 0:
            minority_upsampled = minority_class.sample(len(X[y == 0]), replace=True, random_state=random_state)
            X_balanced = pd.concat([X[y == 0], minority_upsampled], axis=0)
            y_balanced = pd.concat([y[y == 0], pd.Series([1] * len(minority_upsampled))], axis=0)
        else:
            X_balanced = X.copy()
            y_balanced = y.copy()
    
    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)

# ---------------------------
# Enhanced Models (Only using available libraries)
# ---------------------------
# Random Forest with tuned parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Logistic Regression with regularization
log_reg_model = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    class_weight='balanced',
    random_state=42,
    C=0.1,
    penalty='l2'
)

# Enhanced MLP
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

# Gradient Boosting (from sklearn)
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Add XGBoost and LightGBM if available
if XGB_AVAILABLE:
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

if LGB_AVAILABLE:
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

# ---------------------------
# Enhanced Cross Validation
# ---------------------------
def cross_validate_enhanced(X, y, test_data, n_splits=5, n_bags=2, model_name='rf'):
    start_time = datetime.datetime.now()
    scores = []
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(test_data))

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=63)
    
    for fold, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        
        for i in range(n_bags):
            X_train = X.iloc[train_index]
            y_train = y[train_index]
            X_val = X.iloc[valid_index]
            y_val = y[valid_index]

            # Use different sampling strategies
            sampling_method = 'downsample' if i % 2 == 0 else 'upsample'
            X_train_bal, y_train_bal = balanced_sampling(X_train, y_train, method=sampling_method, random_state=10 * fold + i)

            if model_name == 'rf':
                m = clone(rf_model)
            elif model_name == 'logreg':
                m = clone(log_reg_model)
            elif model_name == 'mlp':
                m = clone(mlp_model)
            elif model_name == 'gb':
                m = clone(gb_model)
            elif model_name == 'xgb' and XGB_AVAILABLE:
                m = clone(xgb_model)
            elif model_name == 'lgb' and LGB_AVAILABLE:
                m = clone(lgb_model)
            else:
                continue  # Skip if model not available

            try:
                m.fit(X_train_bal, y_train_bal)
                y_pred = m.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                print(f"Fold {fold+1}, bag {i+1}, {model_name}: ROC-AUC={score:.5f}")
                scores.append(score)

                oof_preds[valid_index] += y_pred / n_bags
                test_preds += m.predict_proba(test_data)[:, 1] / (kfold.get_n_splits() * n_bags)
            except Exception as e:
                print(f"Error in {model_name}, fold {fold+1}, bag {i+1}: {str(e)}")
                continue

            del m
            gc.collect()

    elapsed_time = datetime.datetime.now() - start_time
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\n# {model_name} ROC-AUC: {mean_score:.7f} (+- {std_score:.7f})"
              f" | elapsed: {int(np.round(elapsed_time.total_seconds() / 60))} min")
    else:
        mean_score = 0
        print(f"\n# {model_name} failed to produce any scores")

    return oof_preds, test_preds, mean_score

# ---------------------------
# Advanced Blending with Model Selection
# ---------------------------
def advanced_blend(oof_preds, test_preds, oof_scores, method='score_weighted'):
    """Advanced blending strategies"""
    
    if method == 'rank_weighted':
        # Weight by rank performance
        ranks = rankdata([-score for score in oof_scores])  # Lower rank = better score
        weights = (len(ranks) + 1 - ranks) / sum(len(ranks) + 1 - ranks)
    
    elif method == 'score_weighted':
        # Weight directly by scores
        min_score = min(oof_scores)
        weights = np.array([max(score - min_score + 0.01, 0.01) for score in oof_scores])  # Avoid zero weights
        weights = weights / weights.sum()
    
    elif method == 'equal':
        weights = np.ones(len(oof_preds)) / len(oof_preds)
    
    print(f"Blending weights ({method}): {weights}")
    
    # Blend OOF predictions
    blended_oof = np.zeros_like(oof_preds[0])
    for i, (w, pred) in enumerate(zip(weights, oof_preds)):
        blended_oof += w * pred
    
    # Blend test predictions
    blended_test = np.zeros_like(test_preds[0])
    for i, (w, pred) in enumerate(zip(weights, test_preds)):
        blended_test += w * pred
    
    return blended_oof, blended_test, weights

# ---------------------------
# Feature Importance Analysis
# ---------------------------
def analyze_feature_importance(X, y, top_n=20):
    """Analyze feature importance using Random Forest"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(importances.head(top_n))
    
    return importances.head(top_n)['feature'].tolist()

# ---------------------------
# Main Execution
# ---------------------------
def main():
    # Preprocess data
    print("Preprocessing data...")
    X, preprocessor = preprocess_enhanced(train_data.drop(columns=['id', 'loan_status']))
    y = train_data['loan_status'].values
    test_data_proc, _ = preprocess_enhanced(test_data.drop(columns=['id']))
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    important_features = analyze_feature_importance(X, y)
    
    # Define available models
    models = ['rf', 'logreg', 'gb']  # Base models that are always available
    
    # Add optional models if available
    if XGB_AVAILABLE:
        models.append('xgb')
    if LGB_AVAILABLE:
        models.append('lgb')
    
    # Add MLP cautiously (can be unstable)
    models.append('mlp')
    
    print(f"\nAvailable models: {models}")
    
    oof_predictions = []
    test_predictions = []
    model_scores = []
    model_names = []
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model...")
        print(f"{'='*50}")
        
        oof_pred, test_pred, score = cross_validate_enhanced(
            X, y, test_data_proc, 
            n_splits=5, n_bags=2,
            model_name=model_name
        )
        
        # Only add if we got valid predictions
        if score > 0:
            oof_predictions.append(oof_pred)
            test_predictions.append(test_pred)
            model_scores.append(score)
            model_names.append(model_name)
    
    # Advanced blending
    if oof_predictions:
        print(f"\n{'='*50}")
        print("BLENDING RESULTS")
        print(f"{'='*50}")
        
        blended_oof, blended_test, weights = advanced_blend(
            oof_predictions, test_predictions, model_scores, method='score_weighted'
        )
        
        final_score = roc_auc_score(y, blended_oof)
        print(f"\nFinal Blended OOF ROC-AUC: {final_score:.7f}")
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'loan_status': blended_test
        })
        
        # Apply post-processing calibration
        submission['loan_status'] = np.clip(submission['loan_status'], 0.001, 0.999)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f'submission_enhanced_{final_score:.6f}_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        print(f"\nSubmission saved as: {submission_file}")
        
        print(f"\nModel Performance Summary:")
        for name, score, weight in zip(model_names, model_scores, weights):
            print(f"{name.upper()}: Score = {score:.6f}, Weight = {weight:.4f}")
        
        return final_score, weights
    else:
        print("No models produced valid predictions!")
        return 0, []

if __name__ == "__main__":
    final_score, weights = main()
    if final_score > 0:
        print(f"\nFinal Score: {final_score:.7f}")
    else:
        print("\nTraining failed!")