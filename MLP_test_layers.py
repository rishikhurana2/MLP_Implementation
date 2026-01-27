import numpy as np
import time
from MLP import MLP
import matplotlib.pyplot as plt

# create an MLP where input is two coordinates
# output is class 1 if sum of squares is within unit circle and 0 otherwise    
epochs = 1000

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

neurons_set = [
            [2], # 1 neuron (linear network with softmax)
            [32, 2], # 2 neurons
            [32, 32, 2], # 3 neurons
            [32, 32, 32, 2], # ...
            [32, 32, 32, 32, 2],
            [32, 32, 32, 32, 32, 2],
            [32, 32, 32, 32, 32, 32, 2],
            [32, 32, 32, 32, 32, 32, 32, 2],
          ]

num_layers  = [len(n) for n in neurons_set]
time_ = []
acc   = []
for neurons in neurons_set:
    print(f"Testing {len(neurons)} layers")

    mlp = MLP(input_size=2, hidden_layer_sizes=neurons)

    start_time = time.time()
    loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=0.1)
    train_time = time.time() - start_time
    print(f"time taken to train {len(neurons)} layer MLP is {time.time() - start_time}")

    test_acc = mlp.determine_acc(X=X_val, Y=Y_val)
    time_.append(train_time)
    acc.append(test_acc)

best_layers = neurons_set[np.argmax(acc)]
print(f"Best Validation Acc: {np.max(acc)}")
print(f"Best Layers: {best_layers}")

mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)
# test on the test data
mlp = MLP(input_size=2, hidden_layer_sizes=best_layers)
loss_hist = mlp.train(X=X_train, Y=Y_train, epochs=epochs, eta=0.1, batch_size=len(X_train))
print(f"{len(best_layers)} Layers achieved a test accuracy of: {mlp.determine_acc(X=X_test, Y=Y_test)}")

plt.plot([i for i in range(epochs)], loss_hist)
plt.show()