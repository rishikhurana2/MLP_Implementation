import numpy as np
import matplotlib.pyplot as plt

# MLP where each level is ReLU(Wh_{l - 1} + b_l)
# h_{l - 1} is output of the previous layer
class MLP:
    def __init__(self, input_size, levels, hidden_layer_sizes):
        if levels == 0:
            print("Must have at least one level")
            return
        if len(hidden_layer_sizes) != levels:
            print("Hidden layer size must be as large as levels")
            return

        self.levels = levels        
        self.num_classes = hidden_layer_sizes[-1]        
        self.weights = [] # self.weights[i] holds the parameters for layer i + 1
        self.biases  = []
        prev_size    = input_size

        for lvl in range(levels):
            hidden_layer_size = hidden_layer_sizes[lvl]

            # Using He initialization to prevent convergence to a bad local minimum
            self.weights.append(np.random.normal(size=(hidden_layer_size, prev_size)) * np.sqrt(2/prev_size))
            self.biases.append(np.random.normal(size=(hidden_layer_size,))* np.sqrt(2/prev_size))

            prev_size = hidden_layer_size

    def softmax(self, vec):
        normalize_vec = vec - np.max(vec)
        exp_vec = np.exp(normalize_vec)
        return exp_vec / np.sum(exp_vec)

    def loss_and_grad(self, X, Y):
        # initialize the gradients and biases 
        grad_mats   = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        grad_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]

        loss = 0

        for idx in range(len(X)):
            S,H, _ = self.predict(X[idx])
            logit_scores = S[-1]

            c = Y[idx] # class
            loss += -1 * np.log(self.softmax(logit_scores)[c])
            
            grad_H_l = self.softmax(logit_scores) - np.eye(self.num_classes)[c]
            grad_S_l = grad_H_l # initially, gradient w.r.t S_l equals H_l because top layer does not have ReLU

            # iterate through the levels backwards, doing backprop
            for l in range(self.levels - 1, -1 , -1):
                # if not the last level, then the gradient w.r.t S_l must include gradient of ReLU
                if l < self.levels - 1:
                    grad_S_l = grad_H_l * np.where(S[l] >= 0, 1, 0)                    

                if l >= 1:
                    grad_mats[l] += grad_S_l[:, None] @ H[l - 1][:, None].T
                else:
                    grad_mats[l] += grad_S_l[:, None] @ X[idx][:, None].T

                grad_biases[l] += grad_S_l

                # compute gradient w.r.t output of below layer (for next for loop)
                grad_H_l = (np.sum((grad_S_l * self.weights[l].T).T, axis=0)).T

        loss /= len(X)
        return loss, grad_mats, grad_biases
    
    def train(self, X, Y, epochs, eta=0.01):
        loss_hist = []
        for _ in range(epochs):            
            loss, grad_mats, grad_biases = self.loss_and_grad(X, Y)

            loss_hist.append(loss)

            # gradient descent
            for l in range(self.levels):
                self.weights[l] -= eta * grad_mats[l]
                self.biases[l] -= eta * grad_biases[l]

        return loss_hist

    def predict(self, x):
        inp = x
        H = []
        S = []
        for l in range(self.levels - 1):
            out = self.weights[l] @ inp + self.biases[l]            
            inp = np.maximum(0, out) # input to next layer is ReLU of output

            S.append(out)
            H.append(inp)
        
        # for the last layer, there is no ReLU after, S_L = H_L
        logit_scores = self.weights[-1] @ inp + self.biases[-1]
        S.append(logit_scores)
        H.append(logit_scores)

        # return the maximal value of logit scores vector
        return S, H, np.argmax(self.softmax(logit_scores))

    def predict_no_hidden(self, x):
        _,_, pred = self.predict(x)
        return pred
    
    def determine_acc(self, X, Y):
        model_predict = []
        for x in X:
            model_predict.append(self.predict_no_hidden(x))

        model_predict = np.array(model_predict)
        test_acc = np.sum(model_predict == Y) / len(model_predict)

        return test_acc
    
    # requires training the model first for prediction
    # plots the decision boundary of the trained model
    def plot_decision_boundary(model, X, Y, resolution=200):
        # 1. Define grid
        x_min, x_max = X[:,0].min() - 0.2, X[:,0].max() + 0.2
        y_min, y_max = X[:,1].min() - 0.2, X[:,1].max() + 0.2

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # 2. Flatten grid and run predictions
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        preds = []

        for p in grid_points:
            pred = model.predict_no_hidden(p)
            preds.append(pred)

        preds = np.array(preds).reshape(xx.shape)

        # 3. Plot
        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, preds, alpha=0.4)

        # plot training points
        plt.scatter(X[:,0], X[:,1], c=Y, edgecolor='k')
        plt.title("Decision Boundary")

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.show()
