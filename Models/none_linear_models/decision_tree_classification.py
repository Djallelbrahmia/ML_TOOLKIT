import numpy as np
from Models.utils.tree_data_structure import TreeNode
from Models.utils.criterion import Criterion
from Models.utils.split_dataaset import split_dataset


class DTClassification :
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None  # Racine de l'arbre
    def _best_split(self, X, y):
        """
        Trouve la meilleure caractéristique et le seuil pour diviser les données.
        :param X: Features du dataset.
        :param y: Labels correspondants.
        :return: Indice de la meilleure caractéristique, meilleur seuil, et score associé.
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')  # Minimize l'impureté (indice de Gini ou entropie)
        
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    # Calculer l'impureté pour cette division
                    impurity = self._compute_impurity(y_left, y_right)
                    if impurity < best_score:
                        best_score = impurity
                        best_feature = feature_index
                        best_threshold = threshold
        return best_feature, best_threshold
    def _compute_impurity(self, y_left, y_right):
        """
        Calcule l'impureté totale pour une division.
        :param y_left: Labels de la partie gauche.
        :param y_right: Labels de la partie droite.
        :return: Impureté totale (pondérée).
        """
        total_samples = len(y_left) + len(y_right)
        if self.criterion == 'gini':
            return (len(y_left) / total_samples) * Criterion.gini_index(y_left) + \
                   (len(y_right) / total_samples) * Criterion.gini_index(y_right)
        elif self.criterion == 'entropy':
            return (len(y_left) / total_samples) * Criterion.entropy(y_left) + \
                   (len(y_right) / total_samples) * Criterion.entropy(y_right)
    def fit(self, X, y, depth=0):
        """
        Entraîne l'arbre de décision.
        :param X: Features du dataset.
        :param y: Labels correspondants.
        :param depth: Profondeur actuelle de l'arbre.
        """
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, best_feature, best_threshold)
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        return TreeNode(feature_index=best_feature, threshold=best_threshold,
                        left=left_subtree, right=right_subtree)
    def _calculate_leaf_value(self, y):
        """
        Calcule la valeur prédite pour une feuille (moyenne ou mode).
        :param y: Labels de la feuille.
        :return: Valeur prédite.
        """
        return np.bincount(y).argmax()
    def _traverse_tree(self, x, node):
        """
        Parcourt l'arbre pour faire une prédiction.
        :param x: Un échantillon.
        :param node: Le nœud actuel.
        :return: La prédiction.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """
        Fait des prédictions pour les données entrantes.
        :param X: Features du dataset.
        :return: Prédictions pour chaque échantillon.
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)