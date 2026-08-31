# high_quality_relu_sin.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Training data
X = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
y = np.sin(X)

# 2. ReLU network with more neurons and longer training
model = MLPRegressor(
    hidden_layer_sizes=(200,),   # 200 ReLU neurons
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=20000,
    random_state=0
)
model.fit(X, y.ravel())

# 3. Predictions
x_test = np.linspace(0, 2*np.pi, 800).reshape(-1, 1)
y_pred = model.predict(x_test)

# 4. Plot
plt.figure(figsize=(10, 5))
plt.plot(X, y, label='True sin(x)', color='black', linewidth=2)
plt.plot(x_test, y_pred, color='red', linestyle='--', linewidth=2, label='ReLU NN Approximation')
plt.title('High-Quality ReLU Neural Network Approximation of sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
