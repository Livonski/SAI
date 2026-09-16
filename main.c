#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#define da_appendP(da, e)\
do{\
    if(da->count >= da->capaticy){\
        if(da->capaticy == 0) da->capaticy = 256;\
        else da->capaticy *= 2;\
        da->items = realloc(da->items, da->capaticy * sizeof(*da->items));\
    }\
    da->items[da->count++] = e;\
} while(0)

#define da_init(type) malloc(256 * sizeof(type))

#define layerMax(net, i) i == net.layerIndexes.count - 1 ? (net.neurons.count - 1) : (net.layerIndexes.items[i + 1] - 1)
#define layerMaxP(net, i) i == net->layerIndexes.count - 1 ? (net->neurons.count - 1) : (net->layerIndexes.items[i + 1] - 1)

#define randM11() ((double)rand()/RAND_MAX - 0.5f) * 2
#define randM0505() ((double)rand()/RAND_MAX - 0.5f)

typedef struct{
    int* items;
    int count;
    int capaticy;
} weightIndexes;

typedef struct{
    float value;
    float valueAA;

    float gradient;

    float bias;
    int wIndexStart;
} neuron;

typedef struct{
    neuron* items;
    int count;
    int capaticy;
} neurons;

typedef struct{
    float* items;
    int count;
    int capaticy;
} weights;

typedef struct{
    int* items;
    int count;
    int capaticy;
} layerIndexes;

typedef struct{
    float* items;
    int count;
    int capaticy;
} numbersf;

typedef struct {
    neurons inputLayer;

    neurons neurons;
    layerIndexes layerIndexes;

    neurons outputLayer;

    weights weights;

    numbersf weightGradients;

    
    float prediction;
    float target; 
    float error;

    float loss;

    float outputGradient;
    float dOutput;
} neuralNetwork;

typedef struct{
    float input1;
    float input2;

    float target;
} dataSample;

typedef struct{
    dataSample* items;
    int count;
    int capaticy;
} trainingData;

typedef enum{
    af_ReLu,
    af_sigmoid
}activationFunction;

void GenerateTrainingData(trainingData *tData, int tDataCount){
    //trainingData tData = da_init(dataSample);
    float i1;
    float i2;
    float r;
    
    dataSample s;
    for(int i = 0; i < tDataCount; i++){
        i1 = (double)rand()/RAND_MAX;
        i2 = (double)rand()/RAND_MAX;
        r = i1 + i2 > 1 ? 1.0f : 0.0f;
        s = (dataSample){i1, i2, r};
        da_appendP(tData, s);
    }
}



float relu(float v){
    return v > 0.0f ? v : 0.0f;
}

float sigmoid(float v){
    return 1.0f / (1.0f + expf(-v));
}

float activate(float v, activationFunction func){
    switch(func){
        case 0:
        return relu(v);
        break;
        case 1:
        return sigmoid(v);
        break;
        default:
        assert(false && "Unknown activation function type " && func);
        break;
    }
    return 0;
}

void createRandomNeurons(neuralNetwork *nn, int nCount){
    for(int i = 0; i < nCount; i++){
        float randBias = randM11();
        neuron nNeuron = {.value = 0, .bias = randBias, .wIndexStart = -1};
        da_append(nn->neurons, nNeuron);
    }
}

void GenerateRandomNeuralNetwork(neuralNetwork *nn){
    //input layer
    neuron emptyNeuron = {.value = 1.0f, .valueAA = 1.0f, .bias = 0.0f, .wIndexStart = -1};
    for(int i = 0; i < 2; i++){
        da_append(nn->inputLayer, emptyNeuron);
    }
    
    //inner layers
    da_append(nn->layerIndexes, nn->neurons.count);
    createRandomNeurons(nn, 3);
    
    da_append(nn->layerIndexes, nn->neurons.count);
    createRandomNeurons(nn, 2);
    
    //output layer
    for(int i = 0; i < 1; i++){
        da_append(nn->outputLayer, emptyNeuron);
    }

    //weight construction
    for(int i = 0; i <= nn->layerIndexes.count; i++){
        if(i == 0){
            int layerMax = layerMaxP(nn, i);
            for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
                nn->neurons.items[j].wIndexStart = nn->weights.count;
                for(int k = 0; k < nn->inputLayer.count; k++){
                    printf("Connection - IL(%d) -> Layer(%d:%d)\n", k, i, j);
                    da_append(nn->weights, randM0505());
                }
            }
        } else if (i == nn->layerIndexes.count){
            int layerMax = layerMaxP(nn, i - 1);
            for(int j = 0; j < nn->outputLayer.count; j++){
                nn->outputLayer.items[j].wIndexStart = nn->weights.count;
                for(int k = nn->layerIndexes.items[i - 1]; k <= layerMax; k++){
                    printf("Connection - Layer(%d:%d) -> OL(%d)\n", i - 1, k, j);
                    da_append(nn->weights, randM0505());
                }
            }
        } else {
            int layerMax = layerMaxP(nn, i);
            for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
                int layerMaxPrev = layerMaxP(nn, i - 1);
                nn->neurons.items[j].wIndexStart = nn->weights.count;
                for(int k = nn->layerIndexes.items[i - 1]; k <= layerMaxPrev; k++){
                    printf("Connection - Layer(%d:%d) -> Layer(%d:%d)\n", i - 1, k, i, j);
                    da_append(nn->weights, randM0505());
                }
            }
    
        }
        printf("\n");
    }

    nn->weightGradients.count = nn->weights.count;
    nn->weightGradients.capaticy = nn->weights.capaticy;
    nn->weightGradients.items = malloc(nn->weightGradients.capaticy * sizeof(*nn->weightGradients.items));
}

void neuralNetworkForward(neuralNetwork *nn, dataSample dS){
    nn->inputLayer.items[0].value = dS.input1;
    nn->inputLayer.items[0].valueAA = dS.input1;

    nn->inputLayer.items[1].value = dS.input2;
    nn->inputLayer.items[1].valueAA = dS.input2;

    for(int i = 0; i <= nn->layerIndexes.count; i++){
        if(i == nn->layerIndexes.count){
            for(int j = 0; j < nn->outputLayer.count; j++){
                float value = nn->outputLayer.items[j].bias;
                int cWeight = nn->outputLayer.items[j].wIndexStart;
                int layerMax = layerMaxP(nn, i - 1);
                for(int k = nn->layerIndexes.items[i - 1]; k <= layerMax; k++){
                    value += nn->neurons.items[k].valueAA * nn->weights.items[cWeight];
                    cWeight++;
                }
                nn->outputLayer.items[j].value   = value;
                nn->outputLayer.items[j].valueAA = activate(value, af_sigmoid);
            }
        } else if(i == 0){
            int layerMax = layerMaxP(nn, i);
            for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
                float value = nn->neurons.items[j].bias;
                int cWeight = nn->neurons.items[j].wIndexStart;
                for(int k = 0; k < nn->inputLayer.count; k++){
                    value += nn->inputLayer.items[k].valueAA * nn->weights.items[cWeight];
                    cWeight++;
                }
                nn->neurons.items[j].value   = value;
                nn->neurons.items[j].valueAA = activate(value, af_ReLu);
            }
        }
        else{
            int layerMax = layerMaxP(nn, i);
            for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
                float value = nn->neurons.items[j].bias;
                int cWeight = nn->neurons.items[j].wIndexStart;
                int layerMaxPrev = layerMaxP(nn, i - 1);
                for(int k = nn->layerIndexes.items[i - 1]; k <= layerMaxPrev; k++){
                    value += nn->neurons.items[k].valueAA * nn->weights.items[cWeight];
                    cWeight++;
                }
                nn->neurons.items[j].value   = value;
                nn->neurons.items[j].valueAA = activate(value, af_ReLu);
            }
        }
    }

    
    nn->prediction = nn->outputLayer.items[0].valueAA;
    nn->target     = dS.target; 
    nn->error      = nn->prediction - nn->target;

    nn->loss = nn->error * nn->error;

    nn->outputGradient = 2 * nn->error;
    nn->dOutput        = nn->outputGradient * nn->prediction * (1 - nn->prediction);
}

void neuralNetworkBackward(neuralNetwork *nn){
    //This shit sucks and it's hardcoded as fuck
    //I need to merge input and output layers into the same neuron blob
    
    //Output layer
    for(int j = 0; j < nn->outputLayer.count; j++){
        nn->outputLayer.items[j].gradient = nn->dOutput;

        int cWeight = nn->outputLayer.items[j].wIndexStart;
        int layerMax = layerMaxP(nn, nn->layerIndexes.count - 1);
        for(int k = nn->layerIndexes.items[nn->layerIndexes.count - 1]; k <= layerMax; k++){
            nn->weightGradients.items[cWeight] = nn->outputLayer.items[j].gradient * nn->neurons.items[k].valueAA;
            cWeight++;
        }
    }

    //First layer after output
    int layerMax = layerMaxP(nn, nn->layerIndexes.count - 1);
    int cLayerNStart = nn->layerIndexes.items[nn->layerIndexes.count - 1];
    for(int j = cLayerNStart; j <= layerMax; j++){
        float cGradient = 0;
        for(int k = 0; k < nn->outputLayer.count; k++){
            int cWeight = nn->outputLayer.items[k].wIndexStart + (j - cLayerNStart);
            cGradient += nn->outputLayer.items[k].gradient * nn->weights.items[cWeight];
        }
        cGradient = nn->neurons.items[j].value <= 0.0f ? 0 : cGradient;
        nn->neurons.items[j].gradient = cGradient;

        int layerMaxInner = layerMaxP(nn, nn->layerIndexes.count - 2);
        int cWeight = nn->neurons.items[j].wIndexStart;

        for(int k = nn->layerIndexes.items[nn->layerIndexes.count - 2]; k <= layerMaxInner; k++){
            nn->weightGradients.items[cWeight] = cGradient * nn->neurons.items[k].valueAA;
            cWeight++;
        }
    }

    //last layer
    for(int i = nn-> layerIndexes.count - 2; i >= 0; i--){
        int layerMax = layerMaxP(nn, i);
        for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
            int layerMaxInner = layerMaxP(nn, i + 1);

            float cGradient = 0;
            for(int k = nn->layerIndexes.items[i + 1]; k <= layerMaxInner; k++){
                int cWeight = nn->neurons.items[k].wIndexStart + j - nn->layerIndexes.items[i];
                cGradient += nn->neurons.items[k].gradient * nn->weights.items[cWeight];
            }
            cGradient = nn->neurons.items[j].value <= 0.0f ? 0 : cGradient;
            nn->neurons.items[j].gradient = cGradient;

            int cWeight = nn->neurons.items[j].wIndexStart;
            for(int k = 0; k < nn->inputLayer.count; k++){
                nn->weightGradients.items[cWeight] = cGradient * nn->inputLayer.items[k].valueAA;
                cWeight++;
            }
        }
    }
}

void neuralNetworkUpdate(neuralNetwork *nn, float learningRate){
    for(int i = 0; i < nn->outputLayer.count; i++){
        nn->outputLayer.items[i].bias -= nn->outputLayer.items[i].gradient * learningRate;
    }

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
    printf("--------input layer(%d): \n", nn->inputLayer.count);
    for(int i = 0; i < nn->inputLayer.count; i++){
        printf("----------------|%f Value: %f(%f)\n", nn->inputLayer.items[i].bias, nn->inputLayer.items[i].valueAA, nn->inputLayer.items[i].value);
    }

    for (int i = 0; i < nn->layerIndexes.count; i++){
        int layerMax = i == nn->layerIndexes.count - 1 ? (nn->neurons.count - 1) : (nn->layerIndexes.items[i + 1] - 1);
        printf("------------layer(%d)(%d): \n", i, layerMax);
        for(int j = nn->layerIndexes.items[i]; j <= layerMax; j++){
            printf("----------------|%f Indexes from: %d Value: %f(%f)\n", nn->neurons.items[j].bias, nn->neurons.items[j].wIndexStart, nn->neurons.items[j].valueAA, nn->neurons.items[j].value);
        }
    }

    printf("------------output layer(%d): \n", nn->outputLayer.count);
    for(int i = 0; i < nn->outputLayer.count; i++){
        printf("----------------|%f Indexes from: %d Value: %f(%f)\n", nn->outputLayer.items[i].bias, nn->outputLayer.items[i].wIndexStart, nn->outputLayer.items[i].valueAA, nn->outputLayer.items[i].value);
    }

    printf("----Weights: \n");
    for(int i = 0; i < nn->weights.count; i++){
        printf("--------|%f \n", nn->weights.items[i]);
    }

    printf("Statistics: \n");
    printf("----Prediction: %f\n", nn->outputLayer.items[0].valueAA);
    printf("----Target: %f\n", nn->target);
    printf("----Loss: %f\n", nn->loss);
    printf("----Output gradient: %f\n", nn->outputGradient);
    printf("----Total space occupied by NN: %llub\n", (nn->inputLayer.count + nn->neurons.count + nn->outputLayer.count) * sizeof(float) + nn->weights.count * sizeof(float)); 
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

void neuralNetworkInteractive(neuralNetwork *nn){
    float input1;
    float input2;

    while(true){
        printf("\n");
        printf("Enter two values (or q to quit): ");

        char buffer[256];

        if(fgets(buffer, sizeof(buffer), stdin) == NULL){
            break;
        }

        if(buffer[0] == 'q' || buffer[0] == 'Q'){
            break;
        }

        if(sscanf(buffer, "%f %f", &input1, &input2) != 2){
            printf("Invalid input. Example: 0.25 0.8\n");
            continue;
        }

        dataSample sample = {
            .input1 = input1,
            .input2 = input2,
            .target = 0.0f
        };

        neuralNetworkForward(nn, sample);

        float result = nn->prediction;

        printf("Input:      %.6f %.6f\n", input1, input2);
        printf("Prediction: %.6f\n", result);
        printf("Class:      %d\n", result >= 0.5f ? 1 : 0);
    }

    printf("Interactive mode finished.\n");
}

int main(int argc, char* argv[]){
    float learningRate = 0.01f;
    int numEpochs = 100;

    printf("Generating training data\n");
    trainingData tData = {
        .items = da_init(dataSample),
        .count = 0,
        .capaticy = 256
    };
    GenerateTrainingData(&tData, 1000);
    printf("Generated %d data samples\n", tData.count);

    printf("Generating random neural network\n");
    neuralNetwork nn = {.inputLayer = da_init(neuron), .neurons = da_init(neuron), .layerIndexes = da_init(int), .outputLayer = da_init(neuron), .weights = da_init(float)};
    GenerateRandomNeuralNetwork(&nn);
    printf("Original neural network: \n");
    neuralNetworkForward(&nn, (dataSample){.input1 = 0.0f, .input2 = 0.0f, .target = 0.0f});
    neuralNetworkPrint(&nn);
    
    for(int epoch = 0; epoch < numEpochs; epoch++){
        for(int sample = 0; sample < tData.count; sample++){
            neuralNetworkForward(&nn, tData.items[sample]);
            neuralNetworkBackward(&nn);
            neuralNetworkUpdate(&nn, learningRate);
        }
    }

    printf("Neural network after %d training steps: \n", numEpochs * tData.count);
    neuralNetworkForward(&nn, (dataSample){.input1 = 0.0f, .input2 = 0.0f, .target = 0.0f});
    neuralNetworkPrint(&nn);

    printf("\n--- Interactive mode ---\n");
    neuralNetworkInteractive(&nn);

    return 0;
}