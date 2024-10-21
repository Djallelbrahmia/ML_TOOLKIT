from Models.none_linear_models import NoneLinearRegression
from CostFunctions import MeanSquaredError
import numpy as np

# Example usage
cost_function = MeanSquaredError()  # Assuming you have defined this class
model = NoneLinearRegression(complexity=2, cost_function=cost_function)

# Generate some sample data (X, y)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # Example quadratic relationship

# Fit the model
model.fit(X, y, num_epochs=100, batch_size=5,optimizer="adam", learning_rate=0.01)

# Make predictions
predictions = model.predict(X)
print(predictions)
print("evaluation using MSE")
print(cost_function.compute(predictions=predictions,y=y))  