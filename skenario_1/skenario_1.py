import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_FINAL_DIR = 'data/'
MODEL_FINAL_DIR = 'models/'
RESULT_FINAL_DIR = 'result/'

target = 'islulus'
test = pd.read_csv(os.path.join(DATA_FINAL_DIR, f'train_clean_feature.csv'))
test_lulus = test[test[target] == 1]
test_not_lulus = test[test[target] == 0]
test = pd.concat([test_lulus.sample(
    n=len(test_not_lulus), random_state=12345), test_not_lulus])
test.to_csv(os.path.join(DATA_FINAL_DIR,
            f'test_clean_feature_balanced.csv'), index=False)

# Parameter
dataset = [
    'smote',
    'tvae', 'gaussian_copula',
    'ctgan',
    'ctabgan', 'ctabganplus'
]
n_samples = [20_000, 50_000, 100_000, 250_000, 500_000]
trials = [1, 2, 3, 4, 5]
target = 'islulus'

# Simpan semua hasil
results = []

for ds in dataset:
    for n_sample in n_samples:
        for trial in trials:
            print(
                f'Processing dataset: {ds}, n_samples: {n_sample}, trial: {trial}')
            if n_sample == 500_000 and trial > 1:
                print("Skipping dataset with n_samples=500000 and trial > 1")
                continue

            # Load dataset
            train_path = os.path.join(
                DATA_FINAL_DIR, f'synthetic_train_clean_feature_{ds}.csv')
            test_path = os.path.join(
                DATA_FINAL_DIR, 'test_clean_feature_balanced.csv')

            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            # Sampling
            train = train.sample(n=n_sample, random_state=trial)
            train.replace([np.inf, -np.inf], np.nan, inplace=True)
            train.dropna(inplace=True)

            # Pisahkan fitur dan target
            X_train = train.drop(columns=[target])
            y_train = train[target]
            X_test = test.drop(columns=[target])
            y_test = test[target]

            # Preprocessing: MinMax Scaler
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Buat model Dense NN
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])

            learning_rate = 0.001
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy', metrics=['accuracy'])

            # train_validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=trial)

            # early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)

            # Train model
            history = model.fit(X_train, y_train, epochs=30, validation_data=(
                X_val, y_val), batch_size=64, callbacks=[early_stopping])

            # Save model history
            history_df = pd.DataFrame(history.history)
            history_csv_path = os.path.join(
                MODEL_FINAL_DIR, f'history_clean_feature_{ds}_{n_sample}_{trial}.csv')
            history_df.to_csv(history_csv_path, index=False)
            print(f"Model history saved to: {history_csv_path}")

            # Predict
            y_pred = model.predict(X_test)
            y_pred_labels = (y_pred > 0.5).astype(int)

            # Evaluation
            acc = accuracy_score(y_test, y_pred_labels)
            prec = precision_score(y_test, y_pred_labels)
            rec = recall_score(y_test, y_pred_labels)
            f1 = f1_score(y_test, y_pred_labels)
            cm = confusion_matrix(y_test, y_pred_labels)

            results.append({
                'dataset': ds,
                'n_samples': n_sample,
                'trial': trial,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            })

            model.save(os.path.join(
                MODEL_FINAL_DIR, f'model_clean_feature_{ds}_{n_sample}_{trial}.h5'))

    result_df = pd.DataFrame(results)
    result_df = result_df[result_df['dataset'] == ds]
    result_df.to_csv(os.path.join(
        RESULT_FINAL_DIR, f'result_clean_feature_{ds}.csv'), index=False)

# Simpan ke CSV
result_df = pd.DataFrame(results)
result_csv_path = os.path.join(
    RESULT_FINAL_DIR, f'result_clean_feature.csv')
result_df.to_csv(result_csv_path, index=False)

print(f"Hasil evaluasi disimpan di: {result_csv_path}")
print("Proses selesai.")
