import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from Models.none_linear_models.gaussian_naive_bayes import GaussianNB
from CostFunctions import MeanSquaredError

# Étape 1 : Créer un jeu de données synthétique
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, n_classes=2, random_state=42)

# Étape 2 : Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Étape 3 : Entraîner le modèle GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# Étape 4 : Prédire sur l'ensemble de test
predictions = model.predict(X_test)

# Étape 5 : Calculer l'erreur quadratique moyenne
mse = MeanSquaredError()
mse_value = mse.compute(predictions, y_test)

print(f"Mean Squared Error on the test set: {mse_value}")
