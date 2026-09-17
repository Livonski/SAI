# SAI

SAI is a small neural network implementation written from scratch in C for learning and experimentation.

The project currently trains a simple feed-forward neural network to classify two input values. The target class is `1` when the sum of the inputs is greater than `1`, otherwise it is `0`.

## Features

- Neural network implemented from scratch in C
- Dynamic arrays for neurons, layers and weights
- ReLU and sigmoid activation functions
- Forward propagation
- Backpropagation and gradient descent
- Random training-data generation
- Training statistics and timing
- Interactive mode for testing the trained network

## Current network

```text
Input: 2 neurons
   ↓
Hidden layer: 3 neurons (ReLU)
   ↓
Hidden layer: 2 neurons (ReLU)
   ↓
Output: 1 neuron (Sigmoid)
```

## Build

Using GCC:

```bash
gcc main.c -o main.exe -lm
```

## Run

```bash
./main.exe
```

On Windows Command Prompt:

```cmd
main.exe
```

The program generates training data, creates and trains the neural network, prints training information, and then allows you to enter two values to test the trained model.

Example interactive input:

```text
0.25 0.80
```

Enter `q` to leave interactive mode.

## Purpose

This is an educational project intended to explore how neural networks work internally without using machine-learning libraries.
