import numpy as np

class NeuralNet:

    def __init__(self, layers, optimizer="stochastic"):
        # layers = [inputs, hiddenNeurons, output]
        self.numLayers = len(layers)
        self.sizes = layers
        # y numNeurons in next layer
        # x number of inputs from previous layer
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        # apply random bias term to current neurons
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.optimizer = optimizer  # Choose between 'batch' or 'stochastic'

    def sigmoidFunction(self, weightedInput):
        return 1 / (1 + np.exp(-weightedInput))

    def sigmoidPrimeFunction(self, weightedInput):
        s = self.sigmoidFunction(weightedInput)
        return s * (1 - s)

    def feedForward(self, inputLayer):
        # Initialize activation list
        self.activations = [inputLayer]
        self.weightedInputValues = []

        for i in range(self.numLayers - 1):
            # Calculate weighted input
            weightedInput = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.weightedInputValues.append(weightedInput)

            # Activation function
            activation = self.sigmoidFunction(weightedInput)
            self.activations.append(activation)

        return self.activations[-1], self.activations

    def mse(self, outcome, predicted):
        return 0.5 * np.linalg.norm(outcome - predicted)**2

    def backPropagation(self, inputLayer, expectedOutput):
        output, activations = self.feedForward(inputLayer)

        gradientBias = [np.zeros(bias.shape) for bias in self.biases]
        gradientWeights = [np.zeros(weight.shape) for weight in self.weights]

        # Calculate the error from the output layer
        delta = (output - expectedOutput) * self.sigmoidPrimeFunction(self.weightedInputValues[-1])

        gradientBias[-1] = delta
        gradientWeights[-1] = np.dot(delta, self.activations[-2].T)

        # Now loop backward through the rest of the layers
        for layer in range(2, self.numLayers):
            z = self.weightedInputValues[-layer]
            sp = self.sigmoidPrimeFunction(z)
            delta = np.dot(self.weights[-layer+1].T, delta) * sp
            gradientBias[-layer] = delta
            gradientWeights[-layer] = np.dot(delta, self.activations[-layer-1].T)

        return gradientBias, gradientWeights

    def updateWeightAndBias(self, inputLayer, expectedOutput, learningRate):
        gradientBiases, gradientWeights = self.backPropagation(inputLayer, expectedOutput)

        oldWeights = [w.copy() for w in self.weights]
        oldBiases = [b.copy() for b in self.biases]

        # Update weights and biases
        self.weights = [w - learningRate * gw for w, gw in zip(self.weights, gradientWeights)]
        self.biases = [b - learningRate * gb for b, gb in zip(self.biases, gradientBiases)]

        # Visualize changes in weights and biases
        for i, (old, new) in enumerate(zip(oldWeights, self.weights)):
            diff = new - old
            print(f"Layer {i} Weight Change (norm): {np.linalg.norm(diff)}")

        for i, (old, new) in enumerate(zip(oldBiases, self.biases)):
            diff = new - old
            print(f"Layer {i} Bias Change (norm): {np.linalg.norm(diff)}")

        return self.weights, self.biases

    def train(self, inputs, expectedOutputs, learningRate=0.1, epochs=100, batch_size=None):
        # Batch Gradient Descent or Stochastic Gradient Descent
        losses = []
        for epoch in range(epochs):
            if self.optimizer == "batch":
                # Compute gradients using all data points in the batch
                total_gradient_weights = [np.zeros_like(w) for w in self.weights]
                total_gradient_biases = [np.zeros_like(b) for b in self.biases]
                total_loss = 0

                for i in range(len(inputs)):
                    inputLayer = inputs[i]
                    expectedOutput = expectedOutputs[i]
                    gradientBiases, gradientWeights = self.backPropagation(inputLayer, expectedOutput)
                    
                    # Sum up the gradients
                    total_gradient_biases = [total_grad + grad for total_grad, grad in zip(total_gradient_biases, gradientBiases)]
                    total_gradient_weights = [total_grad + grad for total_grad, grad in zip(total_gradient_weights, gradientWeights)]

                    # Calculate total loss
                    output, _ = self.feedForward(inputLayer)
                    total_loss += self.mse(expectedOutput, output)

                # Update weights and biases after going through all examples (batch update)
                self.weights = [w - learningRate * grad_w / len(inputs) for w, grad_w in zip(self.weights, total_gradient_weights)]
                self.biases = [b - learningRate * grad_b / len(inputs) for b, grad_b in zip(self.biases, total_gradient_biases)]
                losses.append(total_loss / len(inputs))

            elif self.optimizer == "stochastic":
                # Stochastic Gradient Descent - update weights for each data point
                for i in range(len(inputs)):
                    inputLayer = inputs[i]
                    expectedOutput = expectedOutputs[i]
                    self.updateWeightAndBias(inputLayer, expectedOutput, learningRate)
                    output, _ = self.feedForward(inputLayer)
                    loss = self.mse(expectedOutput, output)
                    losses.append(loss)

        return losses

    #def seeStuff(self):
        #print(f" Num Layers: {self.numLayers}")
        #print(f" Layers: {self.sizes}")
        #print(f" Weights: {self.weights}")
        #print(f" Biases: {self.biases}")
        #print(f" Weighted Inputs: {self.weightedInputValues}")
        #print(f" Activations: {self.activations}")

