#include "ReLU.h"

ReLU::ReLU() = default;

Vector &ReLU::forwardPropagation(Vector &input) {
    this->input = input;
    output = input.relu();
    return output;
}

Vector ReLU::backPropagation(Vector grads, double learningRate) {
    return grads * input.reluPrime();
}
