import numpy as np
from Models.naive_abstract import NaiveBayes


class MultinomialNB(NaiveBayes):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha  # Paramètre de lissage (Laplace smoothing)
        self.feature_probs = None

    def _fit(self, X, y):
        n_features = X.shape[1]
        self.feature_probs = np.zeros((self.n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            # Ajouter le lissage de Laplace (alpha)
            feature_counts = X_c.sum(axis=0) + self.alpha
            # Normaliser pour obtenir les probabilités
            total_counts = feature_counts.sum() 
            self.feature_probs[i, :] = feature_counts / total_counts

    def _calculate_posteriors(self, X):
        posteriors = np.zeros((X.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Calculer le log des probabilités pour éviter les underflows
            log_probs = X.dot(np.log(self.feature_probs[i, :] ))
            posteriors[:, i] = log_probs + np.log(self.class_priors[i])

        return posteriors