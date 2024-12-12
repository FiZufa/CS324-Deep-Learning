import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        self.x = None
        self.params = {'weight': np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features), 
                       'bias': np.zeros((1, out_features))
                       }
        self.grads = {'weight': np.zeros((in_features, out_features)), 
                      'bias': np.zeros((1, out_features))
                      }

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x
        forward_linear = np.dot(x, self.params['weight']) + self.params['bias']
        return forward_linear # dim = M x outfeature

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        # grad wrt to its layer param -> for update param
        self.grads['weight'] = np.dot(self.x.T, dout) # dL/dW = x . dL/dout_layer 
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True) # dL/db = dL/dout_layer
        # grad wrt to layer input -> for prev layer
        dx = np.dot(dout, self.params['weight'].T)
        return dx

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.out = np.maximum(0, x)
        return self.out 

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        return dout * (self.out > 0)
        

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        shift_x = x - np.max(x, axis=1, keepdims=True) # for numerical stability
        exps = np.exp(shift_x)
        exps_out = exps / np.sum(exps, axis=1, keepdims=True) # dim: MxC
        return exps_out 

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout
 
class CrossEntropy(object):

    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        x is the M x C, with each row is sample, and each coolumn is probability for c after softmax forward -> predicted
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        m = y.shape[0] 
        log_likelihood = -np.log(x[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss # scalar

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        x is the M x C, with each row is sample, and each coolumn is probability for c after softmax forward -> predicted
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        grad = x - y
        return grad # MxC
