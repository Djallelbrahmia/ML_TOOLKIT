import numpy as np

class Criterion:
    @staticmethod
    def gini_index(y):
        """
        Calcule l'indice de Gini pour un ensemble de labels.
        :param y: Un tableau de labels.
        :return: L'indice de Gini.
        """
        # Comptage des occurrences de chaque classe
        classes, counts = np.unique(y, return_counts=True)
        probas = counts / counts.sum()
        gini = 1 - np.sum(probas ** 2)
        return gini

    @staticmethod
    def entropy(y):
        """
        Calcule l'entropie pour un ensemble de labels.
        :param y: Un tableau de labels.
        :return: L'entropie.
        """
        # Comptage des occurrences de chaque classe
        classes, counts = np.unique(y, return_counts=True)
        probas = counts / counts.sum()
        # Calcul de l'entropie, en s'assurant de ne pas prendre log(0)
        entropy = -np.sum(probas * np.log2(probas + 1e-9))
        return entropy
