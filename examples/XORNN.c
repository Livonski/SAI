#include "SAI.h"

#include <time.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct{
    numbersf inputs;
    numbersf targets;
} dataSample;

typedef struct{
    //TODO: remove dataSample because storing two big da instead of
    //billion small ones is better for cache locality
    dataSample* items;
    int count;
    int capaticy;

    int testDataStart;
} trainingData;


void GenerateTrainingData(trainingData *tData, int tDataCount){
    const float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    const float targets[4] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    for(int i = 0; i < tDataCount; i++){
        int sampleIndex = i % 4;

        float i1 = inputs[sampleIndex][0];
        float i2 = inputs[sampleIndex][1];
        float r  = targets[sampleIndex];

        dataSample s = {0};

        da_append(s.inputs, i1);
        da_append(s.inputs, i2);
        da_append(s.targets, r);

        da_appendP(tData, s);
    }
}

void neuralNetworkInteractive(neuralNetwork *nn){
    float input1;
    float input2;

    while(true){
        printf("\n");
        printf("Enter two values (1 or 0) (or q to quit): ");

        char buffer[256];

        if(fgets(buffer, sizeof(buffer), stdin) == NULL){
            break;
        }

        if(buffer[0] == 'q' || buffer[0] == 'Q'){
            break;
        }

        if(sscanf(buffer, "%f %f", &input1, &input2) != 2){
            printf("Invalid input. Example: 0.00 1.00\n");
            continue;
        }

        numbersf inputs = {0};
        da_append(inputs, input1);
        da_append(inputs, input2);

        clock_t start = clock();
        neuralNetworkForward(nn, inputs);
        clock_t end = clock();

        float result = nn->predictions.items[0];

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
    int numEpochs = 10000;

    printf("Generating training data\n");
    trainingData tData = {
        .items = NULL,
        .count = 0,
        .capaticy = 0,
        .testDataStart = 4
    };
    GenerateTrainingData(&tData, 8);
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
    
    numbersf zeroedInputs = {0};
    da_append(zeroedInputs, 0.0f);
    da_append(zeroedInputs, 0.0f);
    neuralNetworkForward(&nn, zeroedInputs);
    //neuralNetworkPrint(&nn);

    //Train
    clock_t start = clock();
    for(int epoch = 0; epoch < numEpochs; epoch++){
        for(int sample = 0; sample < tData.testDataStart; sample++){
            neuralNetworkForward(&nn, tData.items[sample].inputs);
            neuralNetworkCalculateGradients(&nn, tData.items[sample].targets);
            neuralNetworkBackward(&nn);
            neuralNetworkUpdate(&nn, learningRate);
        }
    }
    clock_t end = clock();

    //Test
    int totalTestSamples = tData.count - tData.testDataStart;
    int testSamplesPassed = 0;
    for(int sample = tData.testDataStart; sample < tData.count; sample++){
        neuralNetworkForward(&nn, tData.items[sample].inputs);
        int predictedClass = nn.predictions.items[0] >= 0.5f ? 1 : 0;
        int targetClass = tData.items[sample].targets.items[0] >= 0.5f ? 1 : 0;

        if(predictedClass == targetClass){
            testSamplesPassed++;
        }
    }

    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    float accuracy = 100.0f * (float)testSamplesPassed / (float)totalTestSamples;

    printf("Neural network after %d training steps: \n", numEpochs * tData.testDataStart);
    printf("Training time: %.6f seconds\n", elapsedTime);
    printf("Test accuracy: %.2f%%\n", accuracy);
    
    neuralNetworkForward(&nn, zeroedInputs);
    //neuralNetworkPrint(&nn);

    printf("\n--- Interactive mode ---\n");
    neuralNetworkInteractive(&nn);

    //techincly this leaks memory because each data sample contains small da,
    //but this should be perfectly fine
    da_free(tData);

    da_free(layers);

    da_free(nn.neurons);
    da_free(nn.layerIndexes);
    da_free(nn.weights);
    da_free(nn.weightGradients);

    return 0;
}