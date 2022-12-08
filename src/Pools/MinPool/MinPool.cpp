#include "MinPool.h"

MinPool::MinPool(int poolSize, int stride) {
    this->poolSize = poolSize;
    this->stride = stride;
}

Vector &MinPool::forwardPropagation(Vector &input) {
    inputImage = input;

    int outputHeight = (inputImage.dimensions[2] - poolSize) / stride + 1;
    int outputWidth = (inputImage.dimensions[3] - poolSize) / stride + 1;

    int outputDimensions[] = {inputImage.dimensions[0], inputImage.dimensions[1], outputHeight, outputWidth};
    poolingOutput = Vector(4, outputDimensions);
    targets = Vector(4, outputDimensions);

    for (int i = 0; i < inputImage.dimensions[0]; i++) {
        for (int j = 0; j < inputImage.dimensions[1]; j++) {
            for (int k = 0; k < outputDimensions[2]; k++) {
                for (int l = 0; l < outputDimensions[3]; l++) {
                    std::pair<double, int> poolingResult = performPooling(i, j, k, l);
                    double minValue = poolingResult.first;
                    int targetMin = poolingResult.second;

                    poolingOutput.setElementAt(i, j, k, l, minValue);
                    targets.setElementAt(i, j, k, l, (double) targetMin);
                }
            }
        }
    }
    return poolingOutput;
}

std::pair<double, int> MinPool::performPooling(int imageHeightCrawler, int imageWidthCrawler, int outputHeightCrawler, int outputWidthCrawler) {
    auto minValue = (double) INT16_MAX;
    int targetMin = 0;
    int x, y;
    for (int m = 0; m < poolSize; ++m) {
        for (int n = 0; n < poolSize; ++n) {
            x = outputHeightCrawler * stride + m;
            y = outputWidthCrawler * stride + n;

            double value = inputImage.getElementAt(imageHeightCrawler, imageWidthCrawler, x, y);
            if (value > minValue) {
                targetMin = m * poolSize + n;
                minValue = value;
            }
        }
    }
    return std::make_pair(minValue, targetMin);
}

Vector MinPool::backPropagation(Vector grads, double learningRate) {
    Vector inputImageGrads(inputImage.numDimensions, inputImage.dimensions);
    inputImageGrads.initializeZero();

    for(int i = 0; i < inputImage.dimensions[0]; i++){
        for(int j = 0; j < inputImage.dimensions[1]; j++){
            for(int k = 0; k < poolingOutput.dimensions[2]; k++){
                for(int l = 0; l < poolingOutput.dimensions[3]; l++){
                    double gradValue = grads.getElementAt(i, j, k, l);
                    int changedIndex = (int) targets.getElementAt(i, j, k, l);

                    int m = changedIndex / poolSize;
                    int n = changedIndex % poolSize;
                    int x = k * stride + m;
                    int y = l * stride + n;

                    inputImageGrads.setElementAt(i, j, x, y, gradValue);
                }
            }
        }
    }
    return inputImageGrads;
}
