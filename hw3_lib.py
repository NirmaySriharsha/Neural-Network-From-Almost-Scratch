"""a minimal PyTorch like framework for NN"""

# NOTATION
# in the comments below
#     N is batch size.
#     in_features is the input size (not counting the bias)
#     out_features is the output size
#     C is number of classes (10 for this assignment)

# do not import any other libraries
# they are not needed and won't be available to the autograder
import numpy as np

# Do not modify this class
class Module:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *g):
        raise NotImplementedError


def sigmoid(x):
    """
    :param x: np.ndarray
    :return: np.ndarray, same shape as x, element-wise sigmoid of x
    """
    ### TYPE HERE AND REMOVE `pass` below ###
    #pass
    return 1/(1+np.exp(-x))


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.state = None

    def forward(self, x):
        """
        :param x: np.ndarray, shape (N, in_features)
        :return: np.ndarry, elementwise ReLU of input, same shape as x.

        in terms of the writeup,

        this layer computes \vec{z} given \vec{a}.
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        x[x<0] = 0
        self.state = x
        return x

    def backward(self, g):
        """
        :param g: np.ndarray, shape (N, in_features), the gradient of
               loss w.r.t. output of this layer.
        :return: np.ndarray, shape (N, in_features), the gradient of
                 loss w.r.t. input of this layer.
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        dz_dx = self.state
        dz_dx[dz_dx > 0] = 1
        return g*dz_dx




class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        # do not change the variable names below as they would be tested
        self.state = None

    def forward(self, x):
        """
        :param x: np.ndarray, shape (N, in_features)
        :return: np.ndarry, elementwise sigmoid of input, same shape as x.

        in terms of the writeup,

        this layer computes \vec{z} given \vec{a}.

        what
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        self.state = sigmoid(x)
        return self.state

    def backward(self, g):
        """
        :param g: np.ndarray, shape (N, in_features), the gradient of
               loss w.r.t. output of this layer.
        :return: np.ndarray, shape (N, in_features), the gradient of
                 loss w.r.t. input of this layer.
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        return g*(self.state*(1-self.state))



        pass


class Linear(Module):
    def __init__(self, init_params):
        super().__init__()
        init_weights, init_biases = init_params
        # do not modify the name of the following parameters

        # weight has shape (in_features, out_features)
        self.weights = init_weights.copy()
        # bias has shape (1, out_features)
        self.biases = init_biases.copy()
        self.d_weights = None
        self.d_biases = None
        self.in_x = None

    def forward(self, x):
        """input has shape (N, in_features)
        and output has shape (N, out_features)

        in terms of the writeup,

        this layer computes \vec{a} given \vec{x},
        or \vec{b} given \vec{z}.
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        self.in_x = x
        output = np.array(x@self.weights + self.biases)
        return output
        

    def backward(self, g):
        """g is of shape (N, out_features)
        g_input should be of shape (N, in_features)
        Also compute the gradient of weights and biases
        NOTE: the gradient of weights and biases should be the average over this batch
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        self.d_biases = np.array([(g.sum(axis = 0))/g.shape[0]])
        self.d_weights = ((self.in_x.T)@g)/g.shape[0]


        return g@(self.weights.T)


class CrossEntropyLoss(Module):
    """softmax + cross entropy loss"""

    def __init__(self):
        super().__init__()
        # y_hat has a shape of (N, C)
        self.y_hat = None
        # hint: consider converting it to one-hot
        self.y_true = None

    def forward(self, score, label):
        """
        :param score: np.ndarray, shape (N, C), the score values for
               input of softmax ($b_k$ in the writeup).
        :param label: either integer-valued np.ndarray, shape (N,), all in [0,C-1]
               (non-zero idx of $\vec{y}$ in the writeup).
        :return: the mean negative cross entropy loss
               ($J(\alpha, \beta)$ in the write up).
        """
        ### TYPE HERE AND REMOVE `pass` below ###
        #pass
        N, C = score.shape
        #Converting labels to one hot vectors y_true, a matrix of shape (N, C)
        #But this is not actually necessary so I'm commenting it out. 
        y_true = np.zeros((N,C))
        for i in range(N):
            y_true[i, label[i]] = 1
        self.y_true = y_true
        #Calculating Loss
        #self.y_true = label
        #Implementing Softmax
        #exps_normed = np.exp(score - np.max(score, axis = 1, keepdims = True))
        #y_hat = exps_normed/exps_normed.sum(axis = 1, keepdims = True)
        exps = np.exp(score)
        y_hat = exps/exps.sum(axis = 1, keepdims = True)
        self.y_hat = y_hat
        #Calculating Loss
        #relevant_scores = np.zeros(N, )
        #for i in range(N):
        #    relevant_scores[i] = y_hat[i, label[i]]
        #return (-1*np.sum(relevant_scores))/N
        log_y_hat = np.log(y_hat)
        loss = (-1 * (np.sum(y_true*log_y_hat)))
        return (-1*np.sum(y_true*log_y_hat))/N
        

    def backward(self):
        """returns the gradient of loss w.r.t. `score`"""
        ### TYPE HERE AND REMOVE `pass` below ###
        return self.y_hat - self.y_true
        #pass


class GradientDescentOptimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.model.layers:
            if type(layer) is Linear:
                ### TYPE HERE AND REMOVE `pass` below ###
                # update parameters of the linear layer
                #pass
                layer.weights = layer.weights - (self.learning_rate*layer.d_weights)
                layer.biases = layer.biases - (self.learning_rate*layer.d_biases)