#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%% Load and preprocess dataset
def get_preprocessed_data():

    data = pd.read_csv('synthetic_ecommerce_data.csv')
    data = data.sample(n=10000, random_state=42)

    data = data.drop(['Transaction_ID', 'Customer_ID', 'Product_ID', 'Transaction_Date'], axis=1)


    data = pd.get_dummies(data, columns=['Category', 'Region'], drop_first=True)

    scaler = StandardScaler()
    numerical_features = ['Units_Sold', 'Discount_Applied', 'Revenue', 'Clicks', 'Impressions',
                          'Conversion_Rate', 'Ad_CTR', 'Ad_CPC', 'Ad_Spend']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    X = data.drop('Ad_CPC', axis=1).values
    y = data['Ad_CPC'].values

    # Split into training, validation, and test sets
    X_trn, X_temp, y_trn, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_vld, X_tst, y_vld, y_tst = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst

#%% Neural Network Class Definition
class NeuralNetwork:
    def __init__(self, hidden_neurons, output_neurons, alpha, batch_size, epochs=1, seed=0):
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.batch_size = batch_size
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed

        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.trn_error, self.vld_error = [], []

    def fit(self, X, y, X_vld=None, y_vld=None):
        np.random.seed(self.seed)
        self.trn_error = []
        self.vld_error = []

        # Initialize weights and biases
        n_input_features = X.shape[1]
        self.W1, self.b1, self.W2, self.b2 = init_weights(n_input_features, self.hidden_neurons, self.output_neurons)

        # Training loop
        for epoch in range(self.epochs):
            batches = get_batches(len(X), self.batch_size)
            for batch in batches:
                X_batch, y_batch = X[batch], y[batch]
                Z1, A1, Z2, A2 = forward(X_batch, self.W1, self.b1, self.W2, self.b2)

                dW2, db2 = output_layer_grads(X_batch, y_batch, self.W1, Z1, A1, self.W2, Z2, A2)
                dZ2 = A2 - y_batch
                dW1, db1 = hidden_layer_grads(X_batch, y_batch, self.W1, Z1, A1, self.W2, dZ2)

                # Update weights and biases
                self.W2 -= self.alpha * dW2
                self.b2 -= self.alpha * db2
                self.W1 -= self.alpha * dW1
                self.b1 -= self.alpha * db1

            # Track training and validation error
            y_train_pred = self.predict(X)
            trn_rmse = np.sqrt(np.mean((y - y_train_pred) ** 2))
            self.trn_error.append(trn_rmse)

            if X_vld is not None and y_vld is not None:
                y_vld_pred = self.predict(X_vld)
                vld_rmse = np.sqrt(np.mean((y_vld - y_vld_pred) ** 2))
                self.vld_error.append(vld_rmse)

        return self

    def predict(self, X):
        _, _, _, A2 = forward(X, self.W1, self.b1, self.W2, self.b2)
        return A2

#%% Helper Functions
def init_weights(n_input_features, hidden_neurons, output_neurons):
    np.random.seed(0)
    w1 = np.random.randn(n_input_features, hidden_neurons) * 0.01
    b1 = np.zeros((1, hidden_neurons))
    w2 = np.random.randn(hidden_neurons, output_neurons) * 0.01
    b2 = np.zeros((1, output_neurons))
    return w1, b1, w2, b2

def get_batches(n_samples, batch_size):
    return [range(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]

def sigmoid(x):
    print(f"Sigmoid input type: {type(x)}, value: {x}")
    x = np.asarray(x)  # Ensure x is a numpy array
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def forward(X, W1, b1, W2, b2):
    print(f"X shape: {X.shape}, W1 shape: {W1.shape}, b1 shape: {b1.shape}")
    X = np.asarray(X, dtype=np.float64)  # Ensure X is a numpy array
    W1, b1, W2, b2 = map(lambda w: np.asarray(w, dtype=np.float64), (W1, b1, W2, b2))

    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2


def output_layer_grads(X, y, W1, Z1, A1, W2, Z2, A2):
    m = X.shape[0]
    d_z2 = A2 - y
    d_w2 = np.dot(A1.T, d_z2) / m
    db2 = np.sum(d_z2, axis=0, keepdims=True) / m
    return d_w2, db2


def hidden_layer_grads(X, y, W1, Z1, A1, W2, dZ2):
    m = X.shape[0]  # Batch size

    # Compute dZ1 correctly using matrix multiplication
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)  # dZ1 shape will be (32, hidden_neurons)

    # Gradients for W1 and b1
    dW1 = np.dot(X.T, dZ1) / m  # dW1 shape will be (14, hidden_neurons)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # db1 shape will be (1, hidden_neurons)

    return dW1, db1


#%% Train and Evaluate the Model
X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = get_preprocessed_data()

nn = NeuralNetwork(
    hidden_neurons=100,
    output_neurons=1,
    alpha=0.01,
    batch_size=32,
    epochs=150,
    seed=0,
)

nn.fit(X_trn, y_trn, X_vld=X_vld, y_vld=y_vld)

tst_rmse = rmse(y_tst, nn.predict(X_tst))
print(f"Test RMSE: {tst_rmse}")

plt.plot(nn.trn_error, label='Train error')
plt.plot(nn.vld_error, label='Validation error')
plt.title("Learning Curve")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()
