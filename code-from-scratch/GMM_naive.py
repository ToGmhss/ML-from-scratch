import numpy as np
from sklearn.datasets import fetch_openml

class DiagonalGMM:
    def __init__(self, n_components=5, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covars_ = None

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covars_ = np.full((self.n_components, n_features), np.cov(X, rowvar=False).mean())

    def _e_step(self, X):
        probabilities = self._estimate_log_prob(X)
        # Use np.log1p for numerical stability.
        log_prob_norm = np.log1p(np.sum(np.exp(probabilities), axis=1) - 1)
        log_resp = probabilities - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp), log_prob_norm

    def _estimate_log_prob(self, X):
        return -0.5 * (np.sum(np.log(2 * np.pi * self.covars_), axis=1) +
                       np.sum((X[:, np.newaxis] - self.means_) ** 2 / self.covars_, axis=2)) + \
               np.log(self.weights_)

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        weights = np.sum(log_resp, axis=0)
        weighted_X_sum = np.dot(log_resp.T, X)
        inverse_weights = 1 / (weights[:, np.newaxis] + 10 * np.finfo(float).eps)

        self.weights_ = weights / n_samples
        self.means_ = weighted_X_sum * inverse_weights
        self.covars_ = self._estimate_covariances(X, log_resp, weights, inverse_weights)

    def _estimate_covariances(self, X, log_resp, weights, inverse_weights):
        covariances = np.empty((self.n_components, X.shape[1]))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            # Calculate the outer product of diff vectors weighted by the responsibilities
            weighted_diff = log_resp[:, k, np.newaxis] * diff
            # Sum over all samples and divide by the weight to get the covariance matrix
            covariances[k] = np.sum(weighted_diff * diff, axis=0) / weights[k]
        return covariances

    def fit(self, X):
        self._initialize_parameters(X)
        lower_bound = -np.infty

        for n_iter in range(1, self.max_iter + 1):
            log_resp, log_prob_norm = self._e_step(X)
            self._m_step(X, log_resp)
            lower_bound_new = log_prob_norm.mean()

            if abs(lower_bound_new - lower_bound) < self.tol:
                break
            lower_bound = lower_bound_new

    def predict_proba(self, X):
        log_resp, _ = self._e_step(X)
        return log_resp



if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Flatten the images
    X = mnist.data
    y = mnist.target
    #print(X.shape) #(70000, 784)

    # Split the data
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    # Convert labels to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    gmm = DiagonalGMM(n_components=5)
    gmm.fit(X_train)
    prob = gmm.predict_proba(X_test)
    
    # For this example, we'll simply use the maximum probability to determine predicted labels
    predicted_labels = np.argmax(prob, axis=1)

    # Calculate the error rate
    error_rate = np.mean(predicted_labels != y_test)
    print(f"Error rate: {error_rate:.6%}")