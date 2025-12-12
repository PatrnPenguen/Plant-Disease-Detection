import math
import random


def softmax(logits):
    """
    Compute softmax probabilities from raw logits.
    logits: list[float] of length C
    returns: list[float] of length C (sum to 1)
    """
    # Numerical stability: subtract max logit
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    sum_exps = sum(exps)
    if sum_exps == 0.0:
        # Fallback to uniform probabilities
        c = len(logits)
        return [1.0 / c for _ in range(c)]
    return [e / sum_exps for e in exps]


def cross_entropy_loss(probs, true_class):
    """
    Compute cross-entropy loss for one example.
    probs: list[float] of length C (softmax output)
    true_class: int (0..C-1)
    """
    eps = 1e-15
    p = probs[true_class]
    # Clamp probability to avoid log(0)
    p = max(min(p, 1.0 - eps), eps)
    return -math.log(p)


class MulticlassLogisticRegression:
    """
    Simple multiclass logistic regression (softmax regression)
    implemented from scratch using only Python lists.
    """

    def __init__(self, num_features, num_classes, learning_rate=0.1):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Initialize weights and biases with small random values
        # W shape: (num_features, num_classes)
        self.W = [
            [(random.random() - 0.5) * 0.01 for _ in range(num_classes)]
            for _ in range(num_features)
        ]
        # b shape: (num_classes,)
        self.b = [(random.random() - 0.5) * 0.01 for _ in range(num_classes)]

    def _compute_logits(self, x):
        """
        Compute logits (raw scores) for one example.
        x: list[float] of length num_features
        returns: list[float] of length num_classes
        """
        logits = [0.0 for _ in range(self.num_classes)]
        # logits[k] = sum_j x[j] * W[j][k] + b[k]
        for k in range(self.num_classes):
            s = 0.0
            for j in range(self.num_features):
                s += x[j] * self.W[j][k]
            s += self.b[k]
            logits[k] = s
        return logits

    def predict_proba_one(self, x):
        """
        Predict class probabilities for a single example.
        """
        logits = self._compute_logits(x)
        probs = softmax(logits)
        return probs

    def predict_one(self, x):
        """
        Predict class index for a single example.
        """
        probs = self.predict_proba_one(x)
        # argmax
        best_class = 0
        best_prob = probs[0]
        for k in range(1, self.num_classes):
            if probs[k] > best_prob:
                best_prob = probs[k]
                best_class = k
        return best_class

    def predict(self, X):
        """
        Predict class indices for a list of examples.
        X: list of examples, each example is list[float]
        """
        return [self.predict_one(x) for x in X]

    def fit(self, X, y, num_epochs=10, shuffle=True, verbose=True):
        """
        Train the model using stochastic gradient descent.
        X: list of examples, each example is list[float] of length num_features
        y: list[int] of class indices (0..num_classes-1)
        num_epochs: number of passes over the training set
        """
        n_samples = len(X)

        for epoch in range(num_epochs):
            # Optionally shuffle data
            indices = list(range(n_samples))
            if shuffle:
                random.shuffle(indices)

            total_loss = 0.0
            correct = 0

            for idx in indices:
                x = X[idx]
                true_class = y[idx]

                # Forward pass
                logits = self._compute_logits(x)
                probs = softmax(logits)
                loss = cross_entropy_loss(probs, true_class)
                total_loss += loss

                # Accuracy tracking
                pred_class = 0
                best_prob = probs[0]
                for k in range(1, self.num_classes):
                    if probs[k] > best_prob:
                        best_prob = probs[k]
                        pred_class = k
                if pred_class == true_class:
                    correct += 1

                # Backward pass + parameter update (SGD)
                # For each class k:
                # error_k = p_k - 1_{k == true_class}
                for k in range(self.num_classes):
                    if k == true_class:
                        error_k = probs[k] - 1.0
                    else:
                        error_k = probs[k]

                    # Update weights W[j][k]
                    for j in range(self.num_features):
                        grad_w_jk = error_k * x[j]
                        self.W[j][k] -= self.learning_rate * grad_w_jk

                    # Update bias b[k]
                    grad_b_k = error_k
                    self.b[k] -= self.learning_rate * grad_b_k

            avg_loss = total_loss / n_samples
            accuracy = correct / n_samples

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"loss: {avg_loss:.4f} - acc: {accuracy:.4f}"
                )
