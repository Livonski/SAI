#include "SAI.h"

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

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

int main(int argc, char* argv[]){
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
    int testSamplesPassed = neuralNetworkTest(&nn, &tData);

    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    float accuracy = 100.0f * (float)testSamplesPassed / (float)totalTestSamples;

    printf("Neural network after %d training steps: \n", numEpochs * tData.testDataStart);
    printf("Training time: %.6f seconds\n", elapsedTime);
    printf("Test accuracy: %.2f%%\n", accuracy);
    
    neuralNetworkForward(&nn, zeroedInputs);
}