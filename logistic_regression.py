import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)



    def loss(self, X, y=None, reg=0):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_train = X.shape[0]

        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        scores = X.dot(self.W)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores


        # loss
        #############################################################################
        # TODO: Compute the logistic loss and store the loss in loss.               #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        # scores = np.clip(scores, -500, 500)
        # # print("score", scores)
        # probs = 1 / (1 + np.exp(-scores))
        # loss = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)) / num_train
        # loss += reg * np.sum(self.W * self.W) / 2
        scores = np.clip(scores, -500, 500)  # Numerical stability for exp
        probs = 1 / (1 + np.exp(-scores))  # Sigmoid function

        # Ensure numerical stability for log
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        # Reshape y for broadcasting if necessary
        y = y.reshape(-1, 1)

        # Compute the binary cross-entropy loss
        loss = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)) / num_train

        # Add regularization term to the loss
        loss += reg * np.sum(self.W ** 2) / 2


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        dscores = probs - y.reshape(-1, 1)
        dW = X.T.dot(dscores) / num_train
        dW += reg * self.W


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores=X.dot(self.W)
        probs=1/(1+np.exp(-scores))
        y_pred=(probs>0.5).astype(int)


        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

