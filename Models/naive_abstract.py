import numpy as np
from abc import ABC, abstractmethod
from Models.model import Model

class NaiveBayes(Model):
    def __init__(self):
        super().__init__()
        self.classes = None
        self.n_classes = None
        self.epsilon = 1e-10 
        self.class_priors = None

    def fit(self, X, y, optimizer=None, num_epochs=None):
        """
        Les paramètres optimizer et num_epochs sont inclus pour la compatibilité avec la classe Model
        mais ne sont pas utilisés pour Naive Bayes
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.class_priors = np.zeros(self.n_classes)
        
        # Calculer les probabilités a priori des classes
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.mean(y == c)
        
        # Appeler la méthode spécifique à chaque sous-classe
        self._fit(X, y)

    @abstractmethod
    def _fit(self, X, y):
        """Méthode à implémenter par les sous-classes"""
        pass

    def predict(self, X):
        """Prédire les classes pour X"""
        posteriors = self._calculate_posteriors(X)
        return self.classes[np.argmax(posteriors, axis=1)]

    @abstractmethod
    def _calculate_posteriors(self, X):
        """Méthode à implémenter par les sous-classes"""
        pass
    
    def compute_gradients(self, X, y):
        """
        Naive Bayes n'utilise pas de gradients, cette méthode est incluse
        pour la compatibilité avec la classe Model
        """
        raise NotImplementedError("Naive Bayes n'utilise pas de gradients")
