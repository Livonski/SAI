#define _CRT_SECURE_NO_WARNINGS

#include "SAI.h"
#include "drawingGUI.h"

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

uint32_t readBigEndianUInt32(FILE *file)
{
    unsigned char bytes[4];

    if (fread(bytes, 1, 4, file) != 4)
    {
        fprintf(stderr, "Failed to read uint32\n");
        exit(1);
    }

    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8)  |
           ((uint32_t)bytes[3]);
}

void readTrainingData(trainingData *tData, const char *imagesPath, const char *labelsPath){
    
    FILE *f_data = fopen(imagesPath, "rb");

    if (f_data == NULL)
    {
        fprintf(stderr, "Failed to open images file: %s\n", imagesPath);
        exit(1);
    }

    FILE *f_labels = fopen(labelsPath, "rb");

    if (f_labels == NULL)
    {
        fprintf(stderr, "Failed to open labels file: %s\n", labelsPath);
        fclose(f_data);
        exit(1);
    }

    uint32_t d_magic = readBigEndianUInt32(f_data);
    uint32_t imageCount = readBigEndianUInt32(f_data);
    uint32_t rows = readBigEndianUInt32(f_data);
    uint32_t columns = readBigEndianUInt32(f_data);
    
    if (d_magic != 2051)
    {
        fprintf(stderr, "Invalid MNIST image file\n");
        fclose(f_data);
        fclose(f_labels);
        assert(false);
    }

    uint32_t l_magic = readBigEndianUInt32(f_labels);
    uint32_t labelCount = readBigEndianUInt32(f_labels);

    if (l_magic != 2049)
    {
        fprintf(stderr, "Invalid MNIST label file\n");
        fclose(f_labels);
        assert(false);
    }

    if (imageCount != labelCount)
    {
        fprintf(
            stderr,
            "Image count (%u) != label count (%u)\n",
            imageCount,
            labelCount
        );

        fclose(f_data);
        fclose(f_labels);
        assert(false);
    }
    
    printf("MNIST:\n");
    printf("Images: %u\n", imageCount);
    printf("Size: %ux%u\n", columns, rows);
    printf("Labels: %u\n", labelCount);

    size_t pixelCount = (size_t)rows * columns;

    for (uint32_t sampleIndex = 0; sampleIndex < imageCount; sampleIndex++)
    {
        dataSample sample = {0};

        // ----------------------------
        // Input: 784 pixels
        // ----------------------------

        for (size_t pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
        {
            unsigned char pixel;

            if (fread(&pixel, 1, 1, f_data) != 1)
            {
                fprintf(
                    stderr,
                    "Failed to read pixel %zu from image %u\n",
                    pixelIndex,
                    sampleIndex
                );

                fclose(f_data);
                fclose(f_labels);

                exit(1);
            }

            float normalizedPixel = (float)pixel / 255.0f;

            da_append(sample.inputs, normalizedPixel);
        }

        unsigned char label;

        if (fread(&label, 1, 1, f_labels) != 1)
        {
            fprintf(
                stderr,
                "Failed to read label %u\n",
                sampleIndex
            );

            fclose(f_data);
            fclose(f_labels);

            exit(1);
        }

        if (label > 9)
        {
            fprintf(
                stderr,
                "Invalid label %u at sample %u\n",
                label,
                sampleIndex
            );

            fclose(f_data);
            fclose(f_labels);

            exit(1);
        }

        for (int targetIndex = 0; targetIndex < 10; targetIndex++)
        {
            float target =
                targetIndex == label
                ? 1.0f
                : 0.0f;

            da_append(sample.targets, target);
        }

        da_appendP(tData, sample);

        if ((sampleIndex + 1) % 10000 == 0)
        {
            printf(
                "Loaded %u / %u samples\n",
                sampleIndex + 1,
                imageCount
            );
        }
    }

    fclose(f_data);
    fclose(f_labels);

    printf("MNIST loading complete\n");
    printf("Training samples: %d\n", tData->count);
}

void numbersGuiPredict(const float *input, int inputCount, float *outputs, int outputCount, void *userData){
    neuralNetwork *nn = (neuralNetwork *)userData;

    numbersf networkInput = {
        .items = (float *)input,
        .count = inputCount,
        .capaticy = inputCount
    };

    neuralNetworkForward(nn, networkInput);

    int copyCount = nn->predictions.count;

    if(copyCount > outputCount) copyCount = outputCount;

    for(int i = 0; i < copyCount; i++){
        outputs[i] = nn->predictions.items[i];
    }

    for(int i = copyCount; i < outputCount; i++){
        outputs[i] = 0.0f;
    }
}

void numbersGuiPrepareInput(const float *canvas, int canvasWidth, int canvasHeight, float *input, int inputCount, void *userData){
    (void)userData;

    int pixelCount = canvasWidth * canvasHeight;

    assert(pixelCount == inputCount);

    float mass = 0.0f;
    float centerX = 0.0f;
    float centerY = 0.0f;

    for(int y = 0; y < canvasHeight; y++){
        for(int x = 0; x < canvasWidth; x++){
            float value = canvas[y * canvasWidth + x];

            mass += value;

            centerX += value * ((float)x + 0.5f);
            centerY += value * ((float)y + 0.5f);
        }
    }

    memset(input, 0, inputCount * sizeof(float));

    if(mass <= 0.00001f) return;

    centerX /= mass;
    centerY /= mass;

    float desiredCenterX = (float)canvasWidth * 0.5f;
    float desiredCenterY = (float)canvasHeight * 0.5f;

    int offsetX = (int)roundf(desiredCenterX - centerX);
    int offsetY = (int)roundf(desiredCenterY - centerY);

    for(int y = 0; y < canvasHeight; y++){
        for(int x = 0; x < canvasWidth; x++){
            int targetX = x + offsetX;
            int targetY = y + offsetY;

            if(targetX < 0 || targetX >= canvasWidth) continue;
            if(targetY < 0 || targetY >= canvasHeight) continue;

            input[targetY * canvasWidth + targetX] = canvas[y * canvasWidth + x];
        }
    }
}


int main(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    
    const char *digitLabels[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    float learningRate = 0.01f;
    int numEpochs = 10;

    //TODO: read training data
    printf("Generating training data\n");
    trainingData tData = {
        .items = NULL,
        .count = 0,
        .capaticy = 0,
        .testDataStart = 50000
    };
    readTrainingData(&tData,"data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    tData.testDataStart = tData.count;
    readTrainingData(&tData, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    printf("Generating random neural network\n");
    neuralNetwork nn = {0};

    numbers layers = {0};
    //images 28*28 pixels == 784 input neurons
    da_append(layers, 784);
    da_append(layers, 196);
    da_append(layers, 49);
    //10 possible layers
    da_append(layers, 10);

    printf("Original neural network: \n");
    GenerateRandomNeuralNetwork(&nn, layers);

    numbersf zeroedInputs = {0};
    for(int i = 0; i < 784; i++){
        da_append(zeroedInputs, 0.0f);
    }
    neuralNetworkForward(&nn, zeroedInputs);

//Train
    clock_t start = clock();
    neuralNetworkTrain(&nn, &tData, numEpochs, learningRate);
    clock_t end = clock();

    //Test
    int totalTestSamples = tData.count - tData.testDataStart;
    testResults results  = neuralNetworkTestMulticlass(&nn, &tData);

    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    float accuracy = 100.0f * (float)results.testSamplesPassed / (float)totalTestSamples;

    printf("Neural network after %d training steps: \n", numEpochs * tData.testDataStart);
    printf("Training time: %.6f seconds\n", elapsedTime);
    printf("Test accuracy: %.2f%%\n", accuracy);
    printf("Confusion matrix: \n");
    for(int i = 0; i < results.confusingMatrix.count; i++){
        printf("%s = %f passed\n", digitLabels[i], results.confusingMatrix.items[i]);
    }
    

    DrawingGUIConfig gui = {
        .title = "SAI - MNIST",
        .canvasWidth = 28,
        .canvasHeight = 28,
        .cellSize = 18,

        .inputCount = 784,
        .outputCount = 10,

        .brushRadius = 1.7f,

        .autoPredict = true,

        .outputLabels = digitLabels,

        .prepareInput = numbersGuiPrepareInput,
        .predict = numbersGuiPredict,

        .userData = &nn
    };

    DrawingGUI_Run(&gui);

    return 1;
}