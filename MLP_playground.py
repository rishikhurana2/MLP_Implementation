from MLP import MLP
import numpy as np
import matplotlib.pyplot as plt

N_train = 500
X = np.random.uniform(0, 1, size=(N_train, 2))    
Y = np.where(X[:, 0]**2 + X[:, 1]**2 < 1, 1, 0)

X_train = X[:400]
Y_train = Y[:400]

X_val = X[400:]
Y_val = Y[400:]

N_test = 500
X_test = np.random.uniform(0, 1, size=(N_test, 2))
Y_test = np.where(X_test[:, 0]**2 + X_test[:, 1]**2 < 1, 1, 0)

# Empirically best result for the above data:
# neurons = [32, 32, 32, 32, 2]
# Learning Rate = 0.1
# batch_size = 128
# epochs = 300
# momentum = 0.9
# Accuracy of 0.96, with full batch, can achieve 0.98

neurons = [32, 32, 32, 32, 2]
mlp = MLP(input_size=2, hidden_layer_sizes=neurons)

eta = 0.1
batch_size = 128
momentum = 0.9
epochs = 300

loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=eta, batch_size=batch_size, momentum_constant=momentum)
plt.plot([i for i in range(epochs)], loss_hist)
plt.show()

print(mlp.determine_acc(X=X_test, Y=Y_test))

mlp.plot_decision_boundary(X=X_test,Y=Y_test)