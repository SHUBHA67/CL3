import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Simulate Sensor Data
def generate_data(n_samples=100):
    # Healthy: Random normal values
    healthy = np.random.normal(loc=0, scale=1, size=(n_samples, 5))
    # Damaged: Shifted values
    damaged = np.random.normal(loc=3, scale=1, size=(n_samples, 5))
    X = np.vstack((healthy, damaged))
    y = np.array([0]*n_samples + [1]*n_samples)  # 0 = Healthy, 1 = Damaged
    return X, y

# Step 2: Negative Selection Algorithm (NSA) Classifier
class NSAClassifier:
    def __init__(self, n_detectors=200, threshold=2.0):
        self.n_detectors = n_detectors
        self.threshold = threshold
        self.detectors = []

    def _distance(self, x, y):
        return np.linalg.norm(x - y)

    def fit(self, X_self):
        # Generate random detectors and discard those too similar to 'self' (healthy)
        self.detectors = []
        while len(self.detectors) < self.n_detectors:
            detector = np.random.uniform(low=-2, high=5, size=X_self.shape[1])
            if all(self._distance(detector, s) > self.threshold for s in X_self):
                self.detectors.append(detector)
        self.detectors = np.array(self.detectors)

    def predict(self, X):
        preds = []
        for sample in X:
            match = any(self._distance(sample, d) < self.threshold for d in self.detectors)
            preds.append(1 if match else 0)  # 1 = Detected as 'non-self' (damaged)
        return np.array(preds)

# Step 3: Run the Classifier
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train NSA only on healthy samples
X_self = X_train[y_train == 0]
clf = NSAClassifier(n_detectors=200, threshold=2.0)
clf.fit(X_self)
# Predict
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Healthy", "Damaged"]))
