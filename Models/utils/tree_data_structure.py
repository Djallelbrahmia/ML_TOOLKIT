class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Un nœud de l'arbre de décision.
        :param feature_index: L'indice de la caractéristique utilisée pour la division.
        :param threshold: La valeur seuil pour la division.
        :param left: Le sous-arbre gauche.
        :param right: Le sous-arbre droit.
        :param value: La valeur prédite (pour les feuilles).
        """
        self.feature_index = feature_index  # L'indice de la caractéristique de la division
        self.threshold = threshold  # La valeur seuil pour la division
        self.left = left  # Sous-arbre gauche
        self.right = right  # Sous-arbre droit
        self.value = value  # Valeur prédite, uniquement pour les feuilles

    def is_leaf_node(self):
        """
        Vérifie si le nœud est une feuille.
        :return: True si le nœud est une feuille, sinon False.
        """
        return self.value is not None