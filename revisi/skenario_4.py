import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Path setup
DATA_FINAL_DIR = 'data/'
RESULT_FINAL_DIR = 'result/'
os.makedirs(RESULT_FINAL_DIR, exist_ok=True)

# List of models
models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True),
    'LogisticRegression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

target = 'islulus'
k_folds = 5

# Load datasets
train_path = os.path.join(DATA_FINAL_DIR, 'train.csv')
test_path = os.path.join(DATA_FINAL_DIR, 'test.csv')

train = pd.read_csv(train_path)
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train.dropna(inplace=True)
test = pd.read_csv(test_path)

# Preprocessing
X = train.drop(columns=[target])
y = train[target]
X_test = test.drop(columns=[target])
y_test = test[target]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

results = []

for model_name, model in models.items():
    print(f"\nEvaluating model: {model_name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(
            f"  Fold {fold+1} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        results.append({
            'model': model_name,
            'fold': fold + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        })

    # Save per-model result
    model_df = pd.DataFrame([r for r in results if r['model'] == model_name])
    model_df.to_csv(os.path.join(RESULT_FINAL_DIR,
                    f'results_{model_name}.csv'), index=False)

# Save full result
all_df = pd.DataFrame(results)
all_df.to_csv(os.path.join(RESULT_FINAL_DIR, 'result.csv'), index=False)

print("\n✓ Skenario uji machine learning selesai. Hasil disimpan di folder 'result/'.")
