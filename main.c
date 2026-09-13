#include <stdio.h>
#include <stdlib.h>

#include <assert.h>
#include <stdbool.h>

#define da_append(da, e)\
do{\
    if(da.count >= da.capaticy){\
        if(da.capaticy == 0) da.capaticy = 256;\
        else da.capaticy *= 2;\
        da.items = realloc(da.items, da.capaticy * sizeof(*da.items));\
    }\
    da.items[da.count++] = e;\
} while(0)

#define da_init(type) malloc(256 * sizeof(type))

#define layerMax(i) i == nn.layerIndexes.count - 1 ? (nn.neurons.count - 1) : (nn.layerIndexes.items[i + 1] - 1);

typedef struct{
    int* items;
    int count;
    int capaticy;
} weightIndexes;

typedef struct{
    int value;

    int bias;
    int wIndexStart;
    int wIndexEnd; //? not sure if i need it
} neuron;

typedef struct{
    neuron* items;
    int count;
    int capaticy;
} neurons;

typedef struct{
    int* items;
    int count;
    int capaticy;
} weights;

typedef struct{
    int* items;
    int count;
    int capaticy;
} layerIndexes;

typedef struct {
    neurons inputLayer;

    neurons neurons;
    layerIndexes layerIndexes;

    neurons outputLayer;

    weights weights;
} neuralNetwork;

void createRandomNeurons(neuralNetwork *nn, int nCount){
    for(int i = 0; i < nCount; i++){
        // int randBias = rand();
        int randBias = i;
        neuron nNeuron = {.value = 0, .bias = randBias, .wIndexStart = -1, .wIndexEnd = -1};
        da_append(nn->neurons, nNeuron);
    }
}

int main(int argc, char* argv[]){
    neuralNetwork nn = {.inputLayer = da_init(int), .neurons = da_init(neuron), .layerIndexes = da_init(int), .outputLayer = da_init(int), .weights = da_init(int)};
    
    //input layer
    neuron emptyNeuron = {.value = 1, .bias = 0, .wIndexStart = -1, .wIndexEnd = -1};
    for(int i = 0; i < 2; i++){
        da_append(nn.inputLayer, emptyNeuron);
    }
    
    //inner layers
    da_append(nn.layerIndexes, nn.neurons.count);
    createRandomNeurons(&nn, 3);

    da_append(nn.layerIndexes, nn.neurons.count);
    createRandomNeurons(&nn, 2);

    //output layer
    for(int i = 0; i < 1; i++){
        da_append(nn.outputLayer, emptyNeuron);
    }

    //weight construction
    for(int i = 0; i <= nn.layerIndexes.count; i++){
        if(i == 0){
            int layerMax = layerMax(i);
            for(int j = nn.layerIndexes.items[i]; j <= layerMax; j++){
                nn.neurons.items[j].wIndexStart = nn.weights.count;
                for(int k = 0; k < nn.inputLayer.count; k++){
                    printf("Connection - IL(%d) -> Layer(%d:%d)\n", k, i, j);
                    da_append(nn.weights, i + j + k);
                }
                nn.neurons.items[j].wIndexEnd = nn.weights.count;
            }
        } else if (i == nn.layerIndexes.count){
            int layerMax = layerMax(i - 1);
            for(int j = 0; j < nn.outputLayer.count; j++){
                nn.outputLayer.items[j].wIndexStart = nn.weights.count;
                for(int k = nn.layerIndexes.items[i - 1]; k <= layerMax; k++){
                    printf("Connection - Layer(%d:%d) -> OL(%d)\n", i, k, j);
                    da_append(nn.weights, i + j + k);
                }
                nn.outputLayer.items[j].wIndexEnd = nn.weights.count;
            }
        } else {
            int layerMax = layerMax(i)
            for(int j = nn.layerIndexes.items[i]; j <= layerMax; j++){
                int layerMaxPrev = layerMax(i - 1);
                nn.neurons.items[j].wIndexStart = nn.weights.count;
                for(int k = nn.layerIndexes.items[i - 1]; k <= layerMaxPrev; k++){
                    printf("Connection - Layer(%d:%d) -> Layer(%d:%d)\n", i - 1, k, i, j);
                    da_append(nn.weights, i + j + k);
                }
                nn.neurons.items[j].wIndexEnd = nn.weights.count;
            }

        }
        printf("\n");
    }

    //calculating result
    for(int i = 0; i <= nn.layerIndexes.count; i++){
        if(i == nn.layerIndexes.count){
            for(int j = 0; j < nn.outputLayer.count; j++){
                int value = nn.outputLayer.items[j].bias;
                int cWeight = nn.outputLayer.items[j].wIndexStart;
                int layerMax = layerMax(i - 1);
                for(int k = nn.layerIndexes.items[i - 1]; k <= layerMax; k++){
                    value += nn.neurons.items[k].value * nn.weights.items[cWeight];
                    cWeight++;
                }
                nn.outputLayer.items[j].value = value;
            }
        } else if(i == 0){
            int layerMax = layerMax(i);
            for(int j = nn.layerIndexes.items[i]; j <= layerMax; j++){
                int value = nn.neurons.items[j].bias;
                int cWeight = nn.neurons.items[j].wIndexStart;
                for(int k = 0; k < nn.inputLayer.count; k++){
                    value += nn.inputLayer.items[k].value * nn.weights.items[cWeight];
                    cWeight++;
                }
                nn.neurons.items[j].value = value;
            }
        }
        else{
            int layerMax = layerMax(i);
            for(int j = nn.layerIndexes.items[i]; j <= layerMax; j++){
                int value = nn.neurons.items[j].bias;
                int cWeight = nn.neurons.items[j].wIndexStart;
                int layerMaxPrev = layerMax(i - 1);
                for(int k = nn.layerIndexes.items[i - 1]; k <= layerMaxPrev; k++){
                    value += nn.neurons.items[k].value * nn.weights.items[cWeight];
                    cWeight++;
                }
                nn.neurons.items[j].value = value;
            }
        }
    }


    printf("Neural network: \n");
    printf("----Layers: \n");
    printf("--------input layer(%d): \n", nn.inputLayer.count);
    for(int i = 0; i < nn.inputLayer.count; i++){
        printf("----------------%d Value: %d\n", nn.inputLayer.items[i].bias, nn.inputLayer.items[i].value);
    }

    for (int i = 0; i < nn.layerIndexes.count; i++){
        int layerMax = i == nn.layerIndexes.count - 1 ? (nn.neurons.count - 1) : (nn.layerIndexes.items[i + 1] - 1);
        printf("------------layer(%d)(%d): \n", i, layerMax);
        for(int j = nn.layerIndexes.items[i]; j <= layerMax; j++){
            printf("----------------%d Indexes: [%d -> %d] Value: %d\n", nn.neurons.items[j].bias, nn.neurons.items[j].wIndexStart, nn.neurons.items[j].wIndexEnd, nn.neurons.items[j].value);
        }
    }

    printf("------------output layer(%d): \n", nn.outputLayer.count);
    for(int i = 0; i < nn.outputLayer.count; i++){
        printf("----------------%d Indexes: [%d -> %d] Value: %d\n", nn.outputLayer.items[i].bias, nn.outputLayer.items[i].wIndexStart, nn.outputLayer.items[i].wIndexEnd, nn.outputLayer.items[i].value);
    }

    printf("----Weights: \n");
    for(int i = 0; i < nn.weights.count; i++){
        printf("--------%d \n", nn.weights.items[i]);
    }
}