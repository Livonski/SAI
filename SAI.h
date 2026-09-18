#ifndef SAI_H
#define SAI_H

#include <stdlib.h>

/*
    Dynamic arrays
*/

#define da_append(da, e) \
do { \
    if ((da).capaticy == 0) { \
        (da).capaticy = 256; \
        (da).items = malloc((da).capaticy * sizeof(*(da).items)); \
    } \
    if ((da).count >= (da).capaticy) { \
        if ((da).capaticy == 0) (da).capaticy = 256; \
        else (da).capaticy *= 2; \
        (da).items = realloc((da).items, (da).capaticy * sizeof(*(da).items)); \
    } \
    (da).items[(da).count++] = (e); \
} while (0)

#define da_appendP(da, e) \
do { \
    if ((da)->capaticy == 0) { \
        (da)->capaticy = 64; \
        (da)->items = malloc((da)->capaticy * sizeof(*(da)->items)); \
    } \
    if ((da)->count >= (da)->capaticy) { \
        if ((da)->capaticy == 0) (da)->capaticy = 64; \
        else (da)->capaticy *= 2; \
        (da)->items = realloc((da)->items, (da)->capaticy * sizeof(*(da)->items)); \
    } \
    (da)->items[(da)->count++] = (e); \
} while (0)

#define da_init(type) malloc(64 * sizeof(type))

#define da_free(da) \
do { \
    (da).count = 0; \
    (da).capaticy = 0; \
    free((da).items); \
    (da).items = NULL; \
} while (0)

/*
    Activation functions
*/

typedef enum {
    af_none,
    af_ReLu,
    af_sigmoid
} activationFunction;

/*
    Neural network data types
*/

typedef struct {
    float value;
    activationFunction aF;

    float gradient;

    float bias;
    int wIndexStart;
} neuron;

typedef struct {
    neuron *items;
    int count;
    int capaticy;
} neurons;

typedef struct {
    int *items;
    int count;
    int capaticy;
} numbers;

typedef struct {
    float *items;
    int count;
    int capaticy;
} numbersf;

typedef struct {
    neurons neurons;
    numbers layerIndexes;

    numbersf weights;
    numbersf weightGradients;

    numbersf predictions;
    numbersf targets;
    numbersf errors;

    float loss;

    numbersf outputGradients;
    numbersf dOutputs;
} neuralNetwork;

/*
    Layer helpers
*/

#define layerMax(net, i) \
    (((i) == (net).layerIndexes.count - 1) \
        ? ((net).neurons.count - 1) \
        : ((net).layerIndexes.items[(i) + 1] - 1))

#define layerStartP(net, i) \
    ((net)->layerIndexes.items[(i)])

#define layerMaxP(net, i) \
    (((i) == (net)->layerIndexes.count - 1) \
        ? ((net)->neurons.count - 1) \
        : ((net)->layerIndexes.items[(i) + 1] - 1))

/*
    Activation
*/

float relu(float v);
float sigmoid(float v);
float activate(neuron *n);

/*
    Network creation
*/

void createRandomNeurons(
    neuralNetwork *nn,
    int nCount,
    float minBias,
    float maxBias,
    activationFunction aF
);

void GenerateRandomNeuralNetwork(
    neuralNetwork *nn,
    numbers layerSizes
);

/*
    Forward / backward propagation
*/

void neuralNetworkForward(
    neuralNetwork *nn,
    numbersf inputs
);

void neuralNetworkCalculateGradients(
    neuralNetwork *nn,
    numbersf targets
);

void neuralNetworkBackward(
    neuralNetwork *nn
);

void neuralNetworkUpdate(
    neuralNetwork *nn,
    float learningRate
);

/*
    Debugging / statistics
*/

void neuralNetworkPrint(
    neuralNetwork *nn
);

#endif
