import numpy as np
from random import shuffle
from classifier import Classifier


class Softmax(Classifier):
    """A subclass of Classifier that uses the Softmax to classify."""
    def __init__(self, random_seed=0):
        super().__init__('softmax')
        if random_seed:
            np.random.seed(random_seed)

    def loss(self, X, y=None, reg=0):
        scores = None
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_train = X.shape[0]
        num_classes = self.W.shape[1]

        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        scores = np.dot(X, self.W)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        if y is None:
            return scores

        # loss
        #############################################################################
        # TODO: Compute the softmax loss and store the loss in loss.                #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
            # Numeric stability fix by subtracting max from scores
        scores -= np.max(scores, axis=1, keepdims=True)

            # Compute softmax probabilities
        exp_scores = np.exp(scores)
        softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Ensure numerical stability for log
        epsilon = 1e-15
        softmax_probs = np.clip(softmax_probs, epsilon, 1 - epsilon)

            # Check sum of softmax probabilities to ensure they sum to 1
        softmax_sums = np.sum(softmax_probs, axis=1)
        assert np.allclose(softmax_sums, 1), f"Softmax probabilities do not sum to 1: {softmax_sums[:5]}"

            # Compute the loss: average cross-entropy loss and regularization
        correct_log_probs = -np.log(softmax_probs[range(num_train), y])
        data_loss = np.sum(correct_log_probs) / num_train
        reg_loss = 0.5 * reg * np.sum(self.W * self.W)
        loss = data_loss + reg_loss
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################
        dscores = softmax_probs
        dscores[range(num_train), y] -= 1
        dscores /= num_train
        dW = np.dot(X.T, dscores)
        dW += reg * self.W

        # Debug prints
        print("Scores (first 5):", scores[:5])
        print("Softmax probabilities (first 5):", softmax_probs[:5])
        print("Sum of softmax probabilities (first 5):", np.sum(softmax_probs[:5], axis=1))
        print("Correct log probabilities (first 5):", correct_log_probs[:5])
        print("dscores (first 5):", dscores[:5])
        print("Gradient dW (first 5):", dW[:5])
        print("Gradient dW sum:", np.sum(dW))

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW


    def predict(self, X):

        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

