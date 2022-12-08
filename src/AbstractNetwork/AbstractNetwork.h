//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_ABSTRACTNETWORK_H
#define NEURAL_NET_IN_CPP_ABSTRACTNETWORK_H

#include "../Vector/Vector.h"

class AbstractNetwork {
public:
    virtual Vector &forwardPropagation(Vector &input) = 0;

    virtual Vector backPropagation(Vector grads, double learningRate) = 0;

    virtual ~AbstractNetwork() = default;
};

#endif //NEURAL_NET_IN_CPP_ABSTRACTNETWORK_H
