import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_FINAL_DIR = 'data/'
MODEL_FINAL_DIR = 'models/'
RESULT_FINAL_DIR = 'result/'

# Parameter
skenario = [
    '_',
    '_clean_',
    '_feature_',
    '_clean_feature_',
    '_clean_onehot_1_',
    '_clean_onehot_2_',
    '_feature_onehot_1_',
    '_feature_onehot_2_',
    '_clean_feature_onehot_1_',
    '_clean_feature_onehot_2_',
    '_onehot_1_',
    '_onehot_2_'
]

target = 'islulus'
dataset = 'ctabganplus'
k_folds = 5

results = []

for ds in skenario:
    print(f'Processing Skenario: {ds}')

    # Load dataset
    train_path = os.path.join(
        DATA_FINAL_DIR, f'synthetic_train{ds}{dataset}.csv')
    test_path = os.path.join(DATA_FINAL_DIR, f'test{ds}balanced.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.dropna(inplace=True)

    X = train.drop(columns=[target])
    y = train[target]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(test.drop(columns=[target]))
    y_test = test[target]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=12345)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/{k_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val),
                            batch_size=64, callbacks=[early_stopping], verbose=1)

        # Save model history
        history_df = pd.DataFrame(history.history)
        history_csv_path = os.path.join(
            MODEL_FINAL_DIR, f'history_{ds}_fold{fold+1}.csv')
        history_df.to_csv(history_csv_path, index=False)

        # Evaluation on Test Set
        y_pred = model.predict(X_test)
        y_pred_labels = (y_pred > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred_labels)
        prec = precision_score(y_test, y_pred_labels)
        rec = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)
        cm = confusion_matrix(y_test, y_pred_labels)

        results.append({
            'skenario': ds,
            'fold': fold + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        })

        model.save(os.path.join(MODEL_FINAL_DIR,
                   f'model_{ds}_fold{fold+1}.h5'))

# Save results
result_df = pd.DataFrame(results)
result_csv_path = os.path.join(RESULT_FINAL_DIR, 'result.csv')
result_df.to_csv(result_csv_path, index=False)

print(f"Hasil evaluasi disimpan di: {result_csv_path}")
print("Proses selesai.")
