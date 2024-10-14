from Models.linear_models import LinearRegression
from Optimizers import Adam  # or any optimizer you defined
from CostFunctions import MeanSquaredError
import numpy as np

# Create data
X = np.array([[2], [3], [4], [5]])
y = np.array([5, 7, 9, 11])

# Initialize model and optimizer
cost_function=MeanSquaredError()
model = LinearRegression(cost_function=cost_function)
optimizer = Adam(learning_rate=0.01)

# Fit the model
model.fit(X, y, optimizer, num_epochs=30)

# Make predictions
predictions = model.predict(X)
print(predictions)
