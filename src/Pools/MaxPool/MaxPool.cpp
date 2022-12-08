#include "MaxPool.h"

MaxPool::MaxPool(int poolSize, int stride) {
    this->poolSize = poolSize;
    this->stride = stride;
}

Vector &MaxPool::forwardPropagation(Vector &input) {
    inputImage = input;

    int outputHeight = (input.dimensions[2] - poolSize) / stride + 1;
    int outputWidth = (input.dimensions[3] - poolSize) / stride + 1;
    int outputDimensions[] = {input.dimensions[0], input.dimensions[1], outputHeight, outputWidth};
    poolingOutput = Vector(4, outputDimensions);
    
    targets = Vector(4, outputDimensions);
    for (int i = 0; i < input.dimensions[0]; i++) {
        for (int j = 0; j < input.dimensions[1]; j++) {
            for (int k = 0; k < outputDimensions[2]; k++) {
                for (int l = 0; l < outputDimensions[3]; l++) {
                    auto poolingResult = performPooling(i, j, k, l);
                    double maxValue = poolingResult.first;
                    int targetValue = poolingResult.second;
                    poolingOutput.setElementAt(i, j, k, l, maxValue);
                    targets.setElementAt(i, j, k, l, (double) targetValue);
                }
            }
        }
    }

    return poolingOutput;
}

std::pair<double, int> MaxPool::performPooling(int imageHeightCrawler, int imageWidthCrawler, int outputHeightCrawler, int outputWidthCrawler) {
    auto maxValue = (double) INT16_MIN;
    int targetMax = 0;
    int x, y;
    for (int m = 0; m < poolSize; ++m) {
        for (int n = 0; n < poolSize; ++n) {
            x = outputHeightCrawler * stride + m;
            y = outputWidthCrawler * stride + n;

            double value = inputImage.getElementAt(imageHeightCrawler, imageWidthCrawler, x, y);
            if (value > maxValue) {
                targetMax = m * poolSize + n;
                maxValue = value;
            }
        }
    }
    return std::make_pair(maxValue, targetMax);
}

Vector MaxPool::backPropagation(Vector grads, double learningRate) {
    Vector inputGrads(inputImage.numDimensions, inputImage.dimensions);
    inputGrads.initializeZero();

    for (int i = 0; i < inputImage.dimensions[0]; i++) {
        for (int j = 0; j < inputImage.dimensions[1]; j++) {
            for (int k = 0; k < poolingOutput.dimensions[2]; k++) {
                for (int l = 0; l < poolingOutput.dimensions[3]; l++) {
                    double gradValue = grads.getElementAt(i, j, k, l);

                    int index = (int)(targets.getElementAt(i, j, k, l));
                    int m = index / poolSize;
                    int n = index % poolSize;

                    inputGrads.setElementAt(i, j, k * stride + m, l * stride + n, gradValue);
                }
            }
        }
    }
    return inputGrads;
}
