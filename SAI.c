#include "SAI.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

#define randf(min, max) ((min) + ((float)rand() / (float)RAND_MAX) * ((max) - (min)))

float relu(float v){
    return v > 0.0f ? v : 0.0f;
}

float sigmoid(float v){
    return 1.0f / (1.0f + expf(-v));
}

float activate(neuron* n){
    switch(n->aF){
        case af_ReLu:
            return relu(n->value);
            break;
        case af_sigmoid:
            return sigmoid(n->value);
            break;
        case af_none:
            return n->value;
            break;
        default:
            assert(false && "Unknown activation function type " && n->aF);
            break;
    }
    return 0;
}

void createRandomNeurons(neuralNetwork *nn, int nCount, float minBias, float maxBias, activationFunction aF){
    for(int i = 0; i < nCount; i++){
        float randBias = randf(minBias, maxBias);
        neuron nNeuron = {.value = 0, .bias = randBias, .wIndexStart = -1, .aF = aF};
        da_append(nn->neurons, nNeuron);
    }
}

void GenerateRandomNeuralNetwork(neuralNetwork *nn, numbers layerSizes){
    assert(layerSizes.count >= 3 && "SAI supports only networks with at least 3 layers");
    
    //input layer
    da_append(nn->layerIndexes, nn->neurons.count);
    createRandomNeurons(nn, layerSizes.items[0], 0.0f, 0.0f, af_none);

    //inner layers
    for(int i = 1; i < layerSizes.count - 1; i++){
        da_append(nn->layerIndexes, nn->neurons.count);
        createRandomNeurons(nn, layerSizes.items[i], 0.0f, 0.0f, af_ReLu);
    }

    //output layer
    da_append(nn->layerIndexes, nn->neurons.count);
    createRandomNeurons(nn, layerSizes.items[layerSizes.count - 1], 0.0f, 1.0f, af_sigmoid);

    //weight construction
    for(int i = 1; i < nn->layerIndexes.count; i++){
        for(int j = nn->layerIndexes.items[i]; j <= layerMaxP(nn, i); j++){
            int layerMaxPrev = layerMaxP(nn, i - 1);
            nn->neurons.items[j].wIndexStart = nn->weights.count;
            for(int k = nn->layerIndexes.items[i - 1]; k <= layerMaxPrev; k++){
                    printf("Connection - Layer(%d:%d) -> Layer(%d:%d)\n", i - 1, k, i, j);
                    da_append(nn->weights, randf(-0.5f, 0.5f));
            }
        }
    }

    nn->weightGradients.count = nn->weights.count;
    nn->weightGradients.capaticy = nn->weights.capaticy;
    nn->weightGradients.items = malloc(nn->weightGradients.capaticy * sizeof(*nn->weightGradients.items));
}

void neuralNetworkForward(neuralNetwork *nn, numbersf inputs){
    assert((inputs.count == (layerMaxP(nn, 0) + 1)) && "To few inputs");
    
    for(int i = 0; i <= layerMaxP(nn, 0); i++){
        nn->neurons.items[i].value = inputs.items[i];
    }

    for(int i = 1; i < nn->layerIndexes.count; i++){
        for(int j = nn->layerIndexes.items[i]; j <= layerMaxP(nn, i); j++){
            float value = nn->neurons.items[j].bias;
            int cWeight = nn->neurons.items[j].wIndexStart;
            for(int k = nn->layerIndexes.items[i - 1]; k <= layerMaxP(nn, i - 1); k++){
                value += activate(&nn->neurons.items[k]) * nn->weights.items[cWeight];
                cWeight++;
            }
            nn->neurons.items[j].value   = value;
        }
    }

    nn->predictions.count = 0;
    for(int i = layerStartP(nn, nn->layerIndexes.count - 1); i <= layerMaxP(nn, nn->layerIndexes.count - 1); i++){
        da_append(nn->predictions, activate(&nn->neurons.items[i]));
    }
}

void neuralNetworkCalculateGradients(neuralNetwork *nn, numbersf targets){
    nn->targets.count = 0;
    nn->errors.count = 0;
    nn->outputGradients.count = 0;
    nn->dOutputs.count = 0;
    
    float loss = 0.0f;
    for(int i = 0; i < targets.count; i++){
        da_append(nn->targets, targets.items[i]);
        da_append(nn->errors, nn->predictions.items[i] - targets.items[i]);
        loss += nn->errors.items[i] * nn->errors.items[i];
        da_append(nn->outputGradients, 2 * nn->errors.items[i]);
        da_append(nn->dOutputs, nn->outputGradients.items[i] * nn->predictions.items[i] * (1 - nn->predictions.items[i]));
    }
    
    nn->loss = loss;
}

void neuralNetworkBackward(neuralNetwork *nn){
    for(int i = nn->layerIndexes.count - 1; i > 0; i--){
        if(i == nn->layerIndexes.count - 1){
            int oCount = 0;
            for(int j = layerStartP(nn, i); j <= layerMaxP(nn, i); j++){
                nn->neurons.items[j].gradient = nn->dOutputs.items[oCount];
                oCount++;

                int cWeight = nn->neurons.items[j].wIndexStart;
                for(int k = layerStartP(nn, i - 1); k <= layerMaxP(nn, i - 1); k++){
                    nn->weightGradients.items[cWeight] = nn->neurons.items[j].gradient * activate(&nn->neurons.items[k]);
                    cWeight++;
                }
            }
            continue;
        }
        
        for(int j = layerStartP(nn, i); j <= layerMaxP(nn, i); j++){
            float cGradient = 0;
            for(int k = layerStartP(nn, i + 1); k <= layerMaxP(nn, i + 1); k++){
                int cWeight = nn->neurons.items[k].wIndexStart + j - nn->layerIndexes.items[i];
                cGradient += nn->neurons.items[k].gradient * nn->weights.items[cWeight];
            }
            cGradient = nn->neurons.items[j].value <= 0.0f ? 0 : cGradient;
            nn->neurons.items[j].gradient = cGradient;

            int cWeight = nn->neurons.items[j].wIndexStart;
            for(int k = layerStartP(nn, i - 1); k <= layerMaxP(nn, i - 1); k++){
                nn->weightGradients.items[cWeight] = nn->neurons.items[j].gradient * activate(&nn->neurons.items[k]);
                cWeight++;
            }
        }
    }
}

void neuralNetworkUpdate(neuralNetwork *nn, float learningRate){
    for(int i = 0; i < nn->neurons.count; i++){
        nn->neurons.items[i].bias -= nn->neurons.items[i].gradient * learningRate;
    }
    
    for(int i = 0; i < nn->weights.count; i++){
        nn->weights.items[i] -= nn->weightGradients.items[i] * learningRate;
    }
}

void neuralNetworkPrint(neuralNetwork *nn){
    printf("Neural network: \n");
    printf("----Layers: \n");
    for (int i = 0; i < nn->layerIndexes.count; i++){
        int layerMax = i == nn->layerIndexes.count - 1 ? (nn->neurons.count - 1) : (nn->layerIndexes.items[i + 1] - 1);
        printf("------------layer(%d)(%d): \n", i, layerMax);
        for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
            printf("----------------|%f Indexes from: %d Value: %f(%f)\n", nn->neurons.items[j].bias, nn->neurons.items[j].wIndexStart, activate(&nn->neurons.items[j]), nn->neurons.items[j].value);
        }
    }

    printf("----Weights: \n");
    for(int i = 0; i < nn->weights.count; i++){
        printf("--------|%f \n", nn->weights.items[i]);
    }

    printf("Statistics: \n");
    printf("----Predictions:\n");
    for(int i = 0; i < nn->predictions.count; i++){
        printf("--------O(%d): %f\n", i, nn->predictions.items[i]);
    }
    printf("----Targets:\n");
    for(int i = 0; i < nn->targets.count; i++){
        printf("--------O(%d): %f\n", i, nn->targets.items[i]);
    }
    printf("----Loss: %f\n", nn->loss);
    printf("----Output gradients:\n");
    for(int i = 0; i < nn->outputGradients.count; i++){
        printf("--------O(%d): %f\n", i, nn->outputGradients.items[i]);
    }

    printf("----Total space occupied by NN: %llub\n", nn->neurons.count * sizeof(float) + nn->weights.count * sizeof(float)); 
    //Size of neuron counts as float because I only care about bias, value and wStart is not really important

    printf("Gradients: \n");
        printf("----Bias: \n");
    for(int i = 0; i < nn->neurons.count; i++){
        printf("--------|%f \n", nn->neurons.items[i].gradient);
    }
    printf("----Weights: \n");
    for(int i = 0; i < nn->weightGradients.count; i++){
        printf("--------|%f \n", nn->weightGradients.items[i]);
    }
}