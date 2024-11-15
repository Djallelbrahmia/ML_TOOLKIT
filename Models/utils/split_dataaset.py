def split_dataset(X, y, feature_index, threshold):
    """
    Divise le dataset en deux sous-ensembles en fonction d'un critère donné.
    :param X: Features du dataset.
    :param y: Labels correspondants.
    :param feature_index: Indice de la caractéristique à utiliser pour la division.
    :param threshold: Seuil de division.
    :return: Deux sous-ensembles (X_left, y_left), (X_right, y_right).
    """
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return (X[left_indices], y[left_indices]), (X[right_indices], y[right_indices])
