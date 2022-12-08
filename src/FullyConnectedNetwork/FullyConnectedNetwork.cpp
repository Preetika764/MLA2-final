#include <random>
#include "FullyConnectedNetwork.h"

FullyConnectedNetwork::FullyConnectedNetwork(int numInputLayerNodes, int numOutputLayerNodes, int randomSeed) {
    std::default_random_engine generator(randomSeed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int weightDimensions[] = {numInputLayerNodes, numOutputLayerNodes};
    weights = Vector(2, weightDimensions);
    weights.randomInitialization(generator, distribution, sqrt(2.0 / numInputLayerNodes));

    int bias_dims[] = {numOutputLayerNodes};
    bias = Vector(1, bias_dims);
    bias.initializeZero();
}

Vector &FullyConnectedNetwork::forwardPropagation(Vector &input) {
    numInputDimensions = input.numDimensions;
    std::copy(input.dimensions, input.dimensions + input.numDimensions, inputDimensions);
    if (input.numDimensions != 2) {
        int flattenedLayerNumNodes = 1;
        for (int i = 1; i < input.numDimensions; ++i) {
            flattenedLayerNumNodes *= input.dimensions[i];
        }
        int matrixMultipliableDimensions[] = {input.dimensions[0], flattenedLayerNumNodes};
        input.resize(2, matrixMultipliableDimensions);
    }
    inputLayer = input;
    output = input.matrixMultiplication(weights) + bias;
    return output;
}

Vector FullyConnectedNetwork::backPropagation(Vector grads, double learningRate) {
    Vector weightGrads = inputLayer.getTranspose().matrixMultiplication(grads);
    Vector biasGrads = grads.getColumnSums();

    grads = grads.matrixMultiplication(weights.getTranspose());
    grads.resize(numInputDimensions, inputDimensions);

    weights -= weightGrads * learningRate;
    bias -= biasGrads * learningRate;
    return grads;
}

void FullyConnectedNetwork::load(FILE *modelFile) {
    double value;
    for (int i = 0; i < weights.dimensions[0]; ++i) {
        for (int j = 0; j < weights.dimensions[1]; ++j) {
            int read = fscanf(modelFile, "%lf", &value); // NOLINT(cert-err34-c)
            if (read != 1) throw std::runtime_error("Invalid model file");
            weights.setElementAt(i, j, value);
        }
    }

    for (int i = 0; i < bias.dimensions[0]; ++i) {
        int read = fscanf(modelFile, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.setElementAt(i, value);
    }
}

void FullyConnectedNetwork::save(FILE *modelFile) {
    for (int i = 0; i < weights.dimensions[0]; ++i) {
        for (int j = 0; j < weights.dimensions[1]; ++j) {
            fprintf(modelFile, "%.18lf ", weights.getElementAt(i, j));
        }
    }

    for (int i = 0; i < bias.dimensions[0]; ++i) {
        fprintf(modelFile, "%.18lf ", bias.getElementAt(i));
    }
}
