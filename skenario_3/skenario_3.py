import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Bidirectional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_FINAL_DIR = 'data/'
MODEL_FINAL_DIR = 'models/'
RESULT_FINAL_DIR = 'result/'


def build_model(model=None, input_shape=None, optimizer=None, learning_rate=None):

    if model == 'dense':
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    elif model == 'cnn':
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu',
                   input_shape=(input_shape, 1)),
            MaxPooling1D(),
            Dropout(0.2),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(),
            Flatten(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    elif model == 'lstm':
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(input_shape, 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    elif model == 'bilstm':
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True),
                           input_shape=(input_shape, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    elif model == 'gru':
        model = Sequential([
            GRU(128, return_sequences=True,
                           input_shape=(input_shape, 1)),
            Dropout(0.2),
            GRU(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
    elif model == 'bigru':
        model = Sequential([
            Bidirectional(GRU(64, return_sequences=True),
                           input_shape=(input_shape, 1)),
            Dropout(0.2),
            Bidirectional(GRU(32)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


list_model = [
    'dense',
    'cnn',
    'lstm',
    'bilstm',
    'gru',
    'bigru'
]
list_optimizer = ['adam', 'sgd', 'rmsprop']
list_learning_rate = [0.01, 0.001, 0.0001, 0.00001]
list_batch_size = [8, 16, 32, 64, 128]
target = 'islulus'
k_folds = 5

train_path = os.path.join(
    DATA_FINAL_DIR, 'train.csv')
test_path = os.path.join(DATA_FINAL_DIR, 'test.csv')

train = pd.read_csv(train_path)
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train.dropna(inplace=True)
test = pd.read_csv(test_path)

X = train.drop(columns=[target])
y = train[target]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(test.drop(columns=[target]))
y_test = test[target]

kf = KFold(n_splits=k_folds, shuffle=True, random_state=12345)

results = []
for model in tqdm(list_model, desc="Model"):
    for optimizer in tqdm(list_optimizer, desc="  Optimizer", leave=False):
        for learning_rate in tqdm(list_learning_rate, desc="    Learning Rate", leave=False):
            for batch_size in tqdm(list_batch_size, desc="      Batch Size", leave=False):

                for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X)), desc="        Fold", leave=False, total=k_folds):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    input_shape = X_train.shape[1]
                    model_instance = build_model(model=model, input_shape=input_shape,
                                                 optimizer=optimizer, learning_rate=learning_rate)

                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, restore_best_weights=True)

                    history = model_instance.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val),
                                                 batch_size=batch_size, callbacks=[early_stopping], verbose=0)

                    # Save model history
                    history_df = pd.DataFrame(history.history)
                    history_csv_path = os.path.join(
                        MODEL_FINAL_DIR, f'history_{model}_{optimizer}_{learning_rate}_batch{batch_size}_fold{fold+1}.csv')
                    history_df.to_csv(history_csv_path, index=False)

                    # Evaluation on Test Set
                    y_pred = model_instance.predict(X_test)
                    y_pred_labels = (y_pred > 0.5).astype(int)

                    acc = accuracy_score(y_test, y_pred_labels)
                    prec = precision_score(y_test, y_pred_labels)
                    rec = recall_score(y_test, y_pred_labels)
                    f1 = f1_score(y_test, y_pred_labels)
                    cm = confusion_matrix(y_test, y_pred_labels)

                    results.append({
                        'model': model,
                        'optimizer': optimizer,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'fold': fold + 1,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1,
                        'confusion_matrix': cm.tolist()
                    })

                    model_instance.save(os.path.join(MODEL_FINAL_DIR,
                                                     f'model_{model}_{optimizer}_{learning_rate}_batch{batch_size}_fold{fold+1}.keras'))

    result_model_df = pd.DataFrame(results)
    result_model_df = result_model_df[result_model_df['model'] == model]
    result_model_df.to_csv(os.path.join(
        RESULT_FINAL_DIR, f'results_{model}.csv'), index=False)

# Save results
result_df = pd.DataFrame(results)
result_csv_path = os.path.join(RESULT_FINAL_DIR, 'result.csv')
result_df.to_csv(result_csv_path, index=False)
