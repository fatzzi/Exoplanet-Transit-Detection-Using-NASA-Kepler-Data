import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # setting up empty arrays 
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            # get rows for this specific class
            X_c = X[y == c]
            
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            
            pdf_values = self._pdf(idx, x)
            # adding a tiny number to stop log(0) crash
            posterior_class = np.sum(np.log(pdf_values + 1e-9)) 
            
            # add logs instead of multiplying probabilities
            posterior = prior + posterior_class
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9 # epsilon to avoid dividing by zero
        
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# testing the model
if __name__ == "__main__":
    print("Loading data...")
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    print("Training Gaussian Naive Bayes from scratch...")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)

    print("Evaluating on test set...")
    predictions = nb.predict(X_test)

    # calculating metrics manually
    accuracy = np.sum(predictions == y_test) / len(y_test)
    
    # 0 is confirmed planet, 1 is false positive
    TP = np.sum((predictions == 0) & (y_test == 0))
    FP = np.sum((predictions == 0) & (y_test == 1))
    FN = np.sum((predictions == 1) & (y_test == 0))
    TN = np.sum((predictions == 1) & (y_test == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f"\n--- Results ---")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(f"[{TP}]  [{FN}]")
    print(f"[{FP}]  [{TN}]")