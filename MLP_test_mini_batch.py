import numpy as np
import matplotlib.pyplot as plt
import time
from MLP import MLP

epochs = 300

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

best_layers = [32, 32, 32, 32, 2]
mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)

best_eta = 0.01

mini_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, len(X_train)]
accs = []

for batch_size in mini_batch_sizes:
    print(f"Training batch size: {batch_size}")
    mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)

    start_time = time.time()
    mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=best_eta, batch_size=batch_size, adam=True)
    print(f"Train Time (s): {time.time() - start_time}")

    accs.append(mlp.determine_acc(X=X_val, Y=Y_val))

best_batch_size = mini_batch_sizes[np.argmax(accs)]
print(f"Best Validation Acc: {np.max(accs)}")
print(f"Best Mini-Batch Size: {best_batch_size}")

# test on the test data
mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)
loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=best_eta, batch_size=best_batch_size, adam=True)
print(f"{best_batch_size} achieved a test accuracy of: {mlp.determine_acc(X=X_test, Y=Y_test)}")

plt.plot(loss_hist)
plt.show()