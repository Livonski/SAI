#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>

#include <assert.h>
#include <stdbool.h>

#define da_append(da, e)\
do{\
    if(da.capaticy == 0){\
        da.capaticy = 256;\
        da.items = malloc(da.capaticy * sizeof(*da.items));\
    }\
    if(da.count >= da.capaticy){\
        if(da.capaticy == 0) da.capaticy = 256;\
        else da.capaticy *= 2;\
        da.items = realloc(da.items, da.capaticy * sizeof(*da.items));\
    }\
    da.items[da.count++] = e;\
} while(0)

#define da_appendP(da, e)\
do{\
    if(da->capaticy == 0){\
        da->capaticy = 256;\
        da->items = malloc(da->capaticy * sizeof(*da->items));\
    }\
    if(da->count >= da->capaticy){\
        if(da->capaticy == 0) da->capaticy = 256;\
        else da->capaticy *= 2;\
        da->items = realloc(da->items, da->capaticy * sizeof(*da->items));\
    }\
    da->items[da->count++] = e;\
} while(0)

#define da_init(type) malloc(256 * sizeof(type))

#define da_free(da)\
do{\
    da.count = 0;\
    da.capaticy = 0;\
    free(da.items);\
} while(0)

#define layerMax(net, i) (((i) == (net).layerIndexes.count - 1) ? ((net).neurons.count - 1) : ((net).layerIndexes.items[(i) + 1] - 1))

#define layerStartP(net, i) (net->layerIndexes.items[i])
#define layerMaxP(net, i) (((i) == (net)->layerIndexes.count - 1) ? ((net)->neurons.count - 1) : ((net)->layerIndexes.items[(i) + 1] - 1))

#define randM11() ((double)rand()/RAND_MAX - 0.5f) * 2
#define randM0505() ((double)rand()/RAND_MAX - 0.5f)

#define randf(min, max) ((min) + ((float)rand() / (float)RAND_MAX) * ((max) - (min)))

typedef enum{
    af_none,
    af_ReLu,
    af_sigmoid
}activationFunction;

typedef struct{
    int* items;
    int count;
    int capaticy;
} weightIndexes;

typedef struct{
    float value;
    activationFunction aF;

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
    int* items;
    int count;
    int capaticy;
} numbers;

typedef struct{
    float* items;
    int count;
    int capaticy;
} numbersf;

typedef struct {
    neurons neurons;
    layerIndexes layerIndexes;

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


void GenerateTrainingData(trainingData *tData, int tDataCount){
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

void neuralNetworkForward(neuralNetwork *nn, dataSample dS){
    nn->neurons.items[0].value = dS.input1;
    nn->neurons.items[1].value = dS.input2;

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

    nn->prediction = activate(&nn->neurons.items[nn->neurons.count - 1]);
    nn->target     = dS.target; 
    nn->error      = nn->prediction - nn->target;

    nn->loss = nn->error * nn->error;

    nn->outputGradient = 2 * nn->error;
    nn->dOutput        = nn->outputGradient * nn->prediction * (1 - nn->prediction);
}

void neuralNetworkBackward(neuralNetwork *nn){
    for(int i = nn->layerIndexes.count - 1; i > 0; i--){
        if(i == nn->layerIndexes.count - 1){
            for(int j = layerStartP(nn, i); j <= layerMaxP(nn, i); j++){
                nn->neurons.items[j].gradient = nn->dOutput;

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
    printf("----Prediction: %f\n", nn->prediction);
    printf("----Target: %f\n", nn->target);
    printf("----Loss: %f\n", nn->loss);
    printf("----Output gradient: %f\n", nn->outputGradient);
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
        clock_t start = clock();
        neuralNetworkForward(nn, sample);
        clock_t end = clock();

        float result = nn->prediction;

        double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;

        printf("Input:      %.6f %.6f\n", input1, input2);
        printf("Prediction: %.6f\n", result);
        printf("Class:      %d\n", result >= 0.5f ? 1 : 0);
        printf("Thinking time: %.6f seconds\n", elapsedTime);
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
    neuralNetwork nn = {0};
    
    numbers layers = {0};
    da_append(layers, 2);
    da_append(layers, 10);
    da_append(layers, 5);
    da_append(layers, 10);
    da_append(layers, 1);
    
    GenerateRandomNeuralNetwork(&nn, layers);
    printf("Original neural network: \n");
    neuralNetworkForward(&nn, (dataSample){.input1 = 0.0f, .input2 = 0.0f, .target = 0.0f});
    neuralNetworkPrint(&nn);
    clock_t start = clock();
    for(int epoch = 0; epoch < numEpochs; epoch++){
        for(int sample = 0; sample < tData.count; sample++){
            neuralNetworkForward(&nn, tData.items[sample]);
            neuralNetworkBackward(&nn);
            neuralNetworkUpdate(&nn, learningRate);
        }
    }
    clock_t end = clock();

    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Neural network after %d training steps: \n", numEpochs * tData.count);
    printf("Training time: %.6f seconds\n", elapsedTime);
    
    neuralNetworkForward(&nn, (dataSample){.input1 = 0.0f, .input2 = 0.0f, .target = 0.0f});
    neuralNetworkPrint(&nn);

    printf("\n--- Interactive mode ---\n");
    neuralNetworkInteractive(&nn);

    da_free(tData);

    da_free(layers);

    da_free(nn.neurons);
    da_free(nn.layerIndexes);
    da_free(nn.weights);
    da_free(nn.weightGradients);

    return 0;
}