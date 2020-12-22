#Custom Activation Function
#This method works with any sklearn classifier apparently, but not sure how well.
#The "calculation" is correct, but also not sure if the inplace selu derivative functions as intended.
#Should be easy to fix if it doesn't. Need to test.
from sklearn.neural_network.multilayer_perceptron import(ACTIVATIONS, DERIVATIVES, MLPClassifier)
import numpy as np
import math

def selu(X):
        """Compute the scaled exponential linear unit function inplace.
        selu(x) = scale*x if x > 0
        selu(x) = scale * alpha * exp(x)-1 if x <= 0
        where alpha and scale are pre-defined (alpha=1.67326324 and scale=1.05070098).
        """

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # np.clip(X, 0.01, np.finfo(X.dtype).max, out=X)
        X[X>0] *= scale
        X[X<=0] = np.exp(X[X<=0]) * scale * alpha - scale * alpha
        return X

def inplace_selu_derivative(Z, delta):
        """Apply the derivative of the scaled exponential linear unit function.

        Parameters
        ----------
        Z : {array-like, sparse matrix}, shape (n_samples, n_features)
            The data which was output from the rectified linear units activation
            function during the forward pass.

        delta : {array-like}, shape (n_samples, n_features)
             The backpropagated error signal to be modified inplace.
        """

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        temp_Z = Z.copy()
        Z[Z>0] *= Z
        Z[Z<=0] = scale * alpha * np.exp(Z[Z<=0])
        delta = Z.copy()
        Z = temp_Z.copy()

ACTIVATIONS['selu'] = selu
DERIVATIVES['selu'] = inplace_selu_derivative