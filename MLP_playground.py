from MLP import MLP
import numpy as np
import matplotlib.pyplot as plt
import time

N_train = 1000
X = np.random.uniform(0, 1, size=(N_train, 2))    
Y = np.where(X[:, 0]**2 + X[:, 1]**2 < 1, 1, 0)

X_train = X[:900]
Y_train = Y[:900]

X_val = X[900:]
Y_val = Y[900:]

N_test = 500
X_test = np.random.uniform(0, 1, size=(N_test, 2))
Y_test = np.where(X_test[:, 0]**2 + X_test[:, 1]**2 < 1, 1, 0)

# Empirically rlly good result for the above data:
# neurons = [32, 32, 32, 32, 2]
# Learning Rate = 0.001
# batch_size = 32
# epochs = 200
# momentum = 0.9
# Avg Accuracy of ~0.985

neurons = [32, 32, 32, 32, 2]
mlp = MLP(input_size=2, hidden_layer_sizes=neurons)

# Hyperparameters
eta = 0.001
batch_size = 32
momentum = 0.9
epochs = 200

avgAcc = 0.0
numTrain = 10

for _ in range(numTrain):
    mlp = MLP(input_size=2, hidden_layer_sizes=neurons)
    
    loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=eta, batch_size=batch_size, momentum_constant=momentum)

    avgAcc += mlp.determine_acc(X=X_test, Y=Y_test)

avgAcc /= numTrain
print(avgAcc)

plt.plot(loss_hist)
plt.show()
mlp.plot_decision_boundary(X=X_test,Y=Y_test)

# Fast enough to test all pairs and see whats best
# etas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# mini_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, len(X_train)]
# val_accs = {}
# epochs = 300

# for eta in etas:
#     for batch_size in mini_batch_sizes:
#         print(f"Training: (eta={eta}, batch_size={batch_size})")
#         mlp = MLP(input_size=2, hidden_layer_sizes=neurons)

#         start_time = time.time()
#         mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=eta, batch_size=batch_size, momentum_constant=0.9)
#         print(f"Train Time (s): {time.time() - start_time}")

#         val_accs[(eta, batch_size)] = mlp.determine_acc(X=X_val, Y=Y_val)

# keys = list(val_accs.keys())
# values = np.array(list(val_accs.values()))

# maxdex = np.argmax(values)
# best_eta, best_batch_size = keys[maxdex]
# print(f"Best Validation Acc: {np.max(values)}")
# print(f"Best Hyperparameters: (eta={best_eta}, batch_size={best_batch_size})")

# loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=best_eta, batch_size=best_batch_size, momentum_constant=0.9)
# print(f"{best_batch_size} achieved a test accuracy of: {mlp.determine_acc(X=X_test, Y=Y_test)}")

# plt.plot(loss_hist)
# plt.show()

# mlp.plot_decision_boundary(X=X_test,Y=Y_test)