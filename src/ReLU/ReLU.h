//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_RELU_H
#define NEURAL_NET_IN_CPP_RELU_H


#include "../Vector/Vector.h"
#include "../AbstractNetwork/AbstractNetwork.h"

class ReLU : public AbstractNetwork {
    public:
        Vector input;

        Vector output;

        ReLU();

        Vector &forwardPropagation(Vector &input) override;

        Vector backPropagation(Vector grads, double learningRate) override;
};


#endif //NEURAL_NET_IN_CPP_RELU_H
