import numpy as np
import matplotlib.pyplot as plt

# Multi-Layer Perceptron Classifier (softmax). Uses ReLU activation functions
# Functions:
#     Computes loss/grad
#     Trains perceptron given X and Y data (and takes in other training opts). 
#     Contains prediction (forward pass).
#     Contains functions that determine accuracy on data sets with model parameters.
#     Contains function that draws the decesion boundary given X. Y data
class MLP:
    # Initialize the MLP with an input size and an array describing the number of neurons in each layer 
    #
    # @param int        input_size          number of input features to the MLP
    # @param int array  hidden_layer_sizes  Array describing the number of neurons for each level. Last level should be the number of classes.
    def __init__(self, input_size, hidden_layer_sizes):

        if len(hidden_layer_sizes) == 0:
            raise ValueError("Number of levels must be greater than 0")

        self.levels = len(hidden_layer_sizes)        
        self.num_classes = hidden_layer_sizes[-1]        
        self.weights = [] # self.weights[i] holds the parameters for layer i + 1
        self.biases  = [] # self.biases[i] holds the bias for layer i + 1
        prev_size    = input_size

        for lvl in range(self.levels):
            hidden_layer_size = hidden_layer_sizes[lvl]

            # Using He initialization to prevent convergence to a bad local minimum
            self.weights.append(np.random.normal(size=(hidden_layer_size, prev_size)) * np.sqrt(2/prev_size))
            self.biases.append(np.zeros(hidden_layer_size))

            prev_size = hidden_layer_size

    # Compute the softmax of an input vector
    #
    # @param  np.array  vec  an np.array of integers
    # 
    # @return np.array  softmax function applied to vec      
    def softmax(self, vec):
        normalize_vec = vec - np.max(vec)
        exp_vec = np.exp(normalize_vec)
        return exp_vec / np.sum(exp_vec)
     
    # Compute the loss and gradient over X, Y using ReLU activation function for each layer, except the top layer, which uses softmax
    #
    # @param  np.array   X   the feature set to use to compute gradient and loss
    # @param  np.array   Y   the measurement set to use to compute gradient and loss
    #
    # @return np.float   The loss over X,Y
    # @return array      An array of gradients w.r.t to matrix parameters
    # @return array      An array of gradients w.r.t to bias parameters
    def loss_and_grad(self, X, Y):
        # initialize the gradients and biases 
        grad_mats   = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        grad_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]

        loss = 0
        for idx in range(len(X)):
            S,H, _ = self.predict(X[idx])
            logit_scores = S[-1]

            c = Y[idx] # true class
            probs = self.softmax(logit_scores) # class probabilities

            loss += -1 * np.log(probs[c]) # accumulate the cross-entropy loss
            
            # top layer gradient is softmax(logit) - y
            grad_H_l = probs - np.eye(self.num_classes)[c]
            grad_S_l = grad_H_l # initially, gradient w.r.t S_l equals H_l because top layer does not have ReLU

            # iterate through the levels backwards, doing backprop
            for l in range(self.levels - 1, -1 , -1):
                # if not the last level, then the gradient w.r.t S_l must include gradient of ReLU
                if l < self.levels - 1:
                    grad_S_l = grad_H_l * np.where(S[l] >= 0, 1, 0) # need to change this to account for different activation functions            

                if l >= 1:
                    grad_mats[l] += grad_S_l[:, None] @ H[l - 1][:, None].T
                else:
                    grad_mats[l] += grad_S_l[:, None] @ X[idx][:, None].T

                grad_biases[l] += grad_S_l

                # compute gradient w.r.t output of below layer (for next for loop)
                grad_H_l = self.weights[l].T @ grad_S_l

        # normalize the loss and gradients
        loss /= len(X)
        for l in range(self.levels):
            grad_mats[l] /= len(X)
            grad_biases[l] /= len(X)
        
        return loss, grad_mats, grad_biases
    
    # Train MLP on X, Y dataset over input epochs iterations, learning rate eta, batch_size samples per iteration, momentum_constant specifying if and how much momentum to use
    #
    # @param  np.array   X                  The feature set to use to compute gradient and loss
    # @param  np.array   Y                  The measurement set to use to compute gradient and loss
    # @param  int        epochs             The number of iterations for training
    # @param  float      eta                The gradient descent learning rate 
    # @param  int        batch_size         The number of samples to run GD over on each epoch
    # @param  float      momentum_constant  If this is None, then no momentum is used. If a number is specified, specifies the Beta constant in momentum. Recommeded: 0.9
    #
    # @return array      A history of the cross-entropy loss values over all the epochs
    def train(self, X, Y, epochs, eta=0.1, batch_size=None, momentum_constant=None):
        N = len(X) # size of dataset

        if not batch_size: batch_size = N
        
        if not isinstance(batch_size, int):
            raise ValueError("Batch size must be of type integer")

        loss_hist = []

        velocity_mats   = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        velocity_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]
        for _ in range(epochs):
            # random sample mini batch indices
            batch_indices = np.random.choice(np.arange(N), batch_size) 
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]

            loss, grad_mats, grad_biases = self.loss_and_grad(X=X_batch, Y=Y_batch)

            loss_hist.append(loss)

            # gradient descent
            for l in range(self.levels):
                if momentum_constant:
                    # compute velocity
                    velocity_mats[l]   = momentum_constant * velocity_mats[l] + (1 - momentum_constant) * grad_mats[l]
                    velocity_biases[l] = momentum_constant * velocity_biases[l] + (1 - momentum_constant) * grad_biases[l]

                    # move in direction of momentum
                    self.weights[l] -= eta * velocity_mats[l]
                    self.biases[l] -= eta * velocity_biases[l]
                else:
                    # move in direction of gradient descent
                    self.weights[l] -= eta * grad_mats[l]
                    self.biases[l] -= eta * grad_biases[l]                    

        return loss_hist

    # Given an input point x, predict the label
    # 
    # @param  np.array  x  An input data point (must be of size input_size)
    #
    # @return array     An array of all the pre-activation values of each layer
    # @return array     An array of all the post-activation values of each layer (except the last layer)
    # @return np.int    An integer corresponding to the label of the output prediction
    def predict(self, x):
        inp = x
        H = []
        S = []
        for l in range(self.levels - 1):
            out = self.weights[l] @ inp + self.biases[l]            
            inp = np.maximum(0, out) # input to next layer is activation function of output (ReLU)

            S.append(out)
            H.append(inp)
        
        # for the last layer, there is no ReLU after, S_L = H_L
        logit_scores = self.weights[-1] @ inp + self.biases[-1]
        S.append(logit_scores)
        H.append(logit_scores)

        # return the maximal value of logit scores vector
        return S, H, np.argmax(self.softmax(logit_scores))

    # Same as predict() function, but only returns the predicted label and does not return S, H arrays
    #
    # @ param  np.array  x   An input data point (must be of size input_size)
    #
    # @ return np.int    An integer corresponding to the label of the output prediction
    def predict_no_hidden(self, x):
        _,_, pred = self.predict(x)
        return pred
    
    # Determine's the accuracy of the model a test/validation dataset X, Y
    #
    # @param  np.array  X  The X dataset containing the features
    # @param  np.array  Y  The Y dataset contain the ground truth
    #
    # @return np.float  The accuracy of the predictions against the ground truth Y
    def determine_acc(self, X, Y):
        model_predict = []
        for x in X:
            # predict all the data points
            model_predict.append(self.predict_no_hidden(x))

        model_predict = np.array(model_predict)
        # sum all the correct predictions and divide by number of prediction to get the accuracy
        test_acc = np.sum(model_predict == Y) / len(model_predict)

        return test_acc
    
    # Plots the descion boundary using X and Y datasets for visualization. This method only really makes sense for X that has 1 or 2 Dimensional features.
    # I used this function to visualize the performance of my model on prediction labels with X samples from [0,1]^2 and Y labels based on inside/outside the unit circle
    # 
    # @param  np.array  X   The X dataset containing the features
    # @param  np.array  Y   The Y dataset containg  the ground truth
    def plot_decision_boundary(self, X, Y, resolution=200):
        # Define the grid
        x_min, x_max = X[:,0].min() - 0.2, X[:,0].max() + 0.2
        y_min, y_max = X[:,1].min() - 0.2, X[:,1].max() + 0.2

        # Create the grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Flatten grid and run predictions
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        preds = []

        # predict each point in the grid
        for p in grid_points:
            pred = self.predict_no_hidden(p)
            preds.append(pred)

        preds = np.array(preds).reshape(xx.shape)

        # Plot the contour seperating the grid predictions
        plt.figure(figsize=(6,6))
        plt.contourf(xx, yy, preds, alpha=0.4)

        # Visualize the graph contour with the test data
        plt.scatter(X[:,0], X[:,1], c=Y, edgecolor='k')
        plt.title("Decision Boundary")

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.show()