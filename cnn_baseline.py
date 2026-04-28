import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_and_evaluate_cnn():
    print("Loading data...")
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val   = np.load("X_val.npy")
    y_val   = np.load("y_val.npy")
    X_test  = np.load("X_test.npy")
    y_test  = np.load("y_test.npy")
    X_candidates = np.load("X_candidates.npy")


    X_train_cnn      = np.expand_dims(X_train,      axis=2)
    X_val_cnn        = np.expand_dims(X_val,        axis=2)
    X_test_cnn       = np.expand_dims(X_test,       axis=2)
    X_candidates_cnn = np.expand_dims(X_candidates, axis=2)

    n_features = X_train_cnn.shape[1]

    print("Building the Convolutional Neural Network (CNN)...")
    model = Sequential([
        # 1st Convolutional Layer: sliding window of size 3 scanning the features
        Conv1D(filters=32, kernel_size=3, activation='relu',
               input_shape=(n_features, 1)),
        MaxPooling1D(pool_size=2),

        # 2nd Convolutional Layer: finding deeper patterns
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.3),   # randomly turns off 30% of neurons to prevent overfitting

        # High value (>0.5) = FALSE POSITIVE, Low value (<0.5) = CONFIRMED PLANET
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining the CNN (watch the epochs)...")
    history = model.fit(
        X_train_cnn, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val_cnn, y_val),
        verbose=1
    )

    # Testing set evaluation
    print("\nEvaluating on the unseen Test Set...")
    y_pred_probs = model.predict(X_test_cnn).flatten()
    y_pred       = (y_pred_probs > 0.5).astype(int)

    TP = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 0) & (y_test == 1))
    FN = np.sum((y_pred == 1) & (y_test == 0))
    TN = np.sum((y_pred == 1) & (y_test == 1))

    accuracy  = (TP + TN) / len(y_test)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 42)
    print("  CNN BASELINE — TEST SET PERFORMANCE")
    print("=" * 42)
    print(f"Accuracy:  {accuracy  * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall    * 100:.2f}%")
    print(f"F1-Score:  {f1        * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("                 Pred Planet  Pred FP")
    print(f"Actual Planet:   [{TP}]        [{FN}]")
    print(f"Actual FP:       [{FP}]        [{TN}]")
    print("=" * 42)

    print("\nRunning CNN inference on 1,979 candidates...")
    cnn_raw_probs        = model.predict(X_candidates_cnn).flatten()
    cnn_planet_probs     = 1.0 - cnn_raw_probs          # flip: high = planet
    cnn_candidate_labels = (cnn_planet_probs >= 0.5).astype(int)  # 1=planet, 0=FP

    n_planet = np.sum(cnn_candidate_labels == 1)
    n_fp     = np.sum(cnn_candidate_labels == 0)
    print(f"CNN predicts: {n_planet} likely planets, {n_fp} false positives")

    np.save("cnn_candidate_scores.npy",  cnn_planet_probs)
    np.save("cnn_candidate_labels.npy",  cnn_candidate_labels)
    print("Saved: cnn_candidate_scores.npy")
    print("Saved: cnn_candidate_labels.npy")

if __name__ == "__main__":
    build_and_evaluate_cnn()
