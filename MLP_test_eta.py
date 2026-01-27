import numpy as np
import matplotlib.pyplot as plt
import time
from MLP import MLP

epochs = 250

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

best_layers = [32, 32, 32, 32, 32, 32, 32, 32, 32, 2]

etas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
accs = []

for eta in etas:
    print(f"Training LR: {eta}")
    mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)

    start_time = time.time()
    mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=eta, batch_size=len(X_train), momentum=True)
    print(f"Train Time (s): {time.time() - start_time}")

    accs.append(mlp.determine_acc(X=X_val, Y=Y_val))

best_eta = etas[np.argmax(accs)]
print(f"Best Validation Acc: {np.max(accs)}")
print(f"Best Eta: {best_eta}")

# test on the test data
mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)
loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=best_eta, batch_size=len(X_train), momentum=True)
print(f"Eta of {best_eta} achieved a test accuracy of: {mlp.determine_acc(X=X_test, Y=Y_test)}")

plt.plot([i for i in range(epochs)], loss_hist)
plt.show()