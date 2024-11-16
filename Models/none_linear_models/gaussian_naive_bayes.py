import numpy as np
from Models.naive_abstract import NaiveBayes


class GaussianNB(NaiveBayes):
    def __init__(self):
        super().__init__()
        self.theta = None  # Moyennes des caractéristiques par classe
        self.sigma = None  # Variances des caractéristiques par classe
        self.epsilon = 1e-10
    def _fit(self, X, y):
        n_features = X.shape[1]
        self.theta = np.zeros((self.n_classes, n_features))
        self.sigma = np.zeros((self.n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.theta[i, :] = X_c.mean(axis=0)
            self.sigma[i, :] = X_c.var(axis=0) + self.epsilon

    def _calculate_posteriors(self, X):
        posteriors = np.zeros((X.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Calculer la densité gaussienne pour chaque caractéristique
            exponent = -0.5 * ((X - self.theta[i, :]) ** 2) / self.sigma[i, :]
            likelihoods = np.exp(exponent) / np.sqrt(2 * np.pi * self.sigma[i, :])
            # Multiplier les probabilités et ajouter le prior
            posteriors[:, i] = np.sum(np.log(likelihoods + self.epsilon), axis=1)
            posteriors[:, i] += np.log(self.class_priors[i] + self.epsilon)

        return posteriors