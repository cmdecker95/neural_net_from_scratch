# Building a Neural Network from Scratch

Here, for the purpose of deeper understanding, I will build an artificial neural network _from scratch_, meaning without modern machine learning packages (e.g. Scikit-Learn, Tensorflow, and Pytorch). I will use NumPy and good math.

## Part 1: Neural Network with Stochastic Gradient Descent
I will define a class NeuralNetwork that implements an artificial neural network with a single hidden layer. The hidden layer will use the Rectified Linear Unit (ReLU) non-linear activation function, and the output will have a Softmax activation function. This first part will implement SGD (gradient descent with a batch size of 1).

The hardest part of this was the `NeuralNetwork.train()` method, which required computing lots of gradients. Translating these equations into code is non-trivial. Thankfully, the **backpropagation** algorithm is a dynamic programming algorithm that computes the gradients layer by layer, and can be written very elegantly in terms of matrix manipulations.

## Part 2: Apply the Model to Fashion Dataset
I will test the model on the Fashion MNIST dataset. This is a 10-class classification task designed to be similar to the MNIST digit recognition dataset. The classes are different items of clothing (shoes, shirts, handbags, etc.) instead of digits. Here is an [introduction](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) and [github page](https://github.com/zalandoresearch/fashion-mnist).

1. Demonstrate overfitting in the network by plotting the training and test set losses vs. epoch.
2. Optimize the hyperparameters of the neural network.
3. Visualize the 10 test examples with the largest loss.

## Part 3: Better and Faster: Mini-Batch Gradient Descent
Lastly, I will implement mini-batch gradient descent in the `NeuralNetwork.train()` method. It is much more efficient to update the weights after *batches* of training data (e.g. 100 examples at a time), which serves two purposes:
1. Each update is a better, less-noisy estimate of the true gradient.
2. The matrix multiplications can be parallelized for an almost-linear speedup with multiple cores or a GPU. By default, NumPy should automatically use multiple CPUs for matrix multiplications. This requires implementing the forward and backpropagation computations efficiently, using matrix multiplications rather than for loops.