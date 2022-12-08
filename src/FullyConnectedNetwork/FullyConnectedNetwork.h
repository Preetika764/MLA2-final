#ifndef NEURAL_NET_IN_CPP_FULLYCONNECTEDNETWORK_H
#define NEURAL_NET_IN_CPP_FULLYCONNECTEDNETWORK_H

#include "../AbstractNetwork/AbstractNetwork.h"
#include "../Loadable/Loadable.h"
#include "../Vector/Vector.h"

class FullyConnectedNetwork : public AbstractNetwork, public Loadable {
public:
    int numInputDimensions;

    int inputDimensions[4];

    Vector inputLayer;

    Vector weights;

    Vector bias;

    Vector output;

    FullyConnectedNetwork(int numInputLayerNodes, int numOutputLayerNodes, int randomSeed = 0);

    Vector &forwardPropagation(Vector &input) override;

    Vector backPropagation(Vector grads, double learningRate) override;

    void load(FILE *modelFile) override;

    void save(FILE *modelFile) override;
};


#endif //NEURAL_NET_IN_CPP_FULLYCONNECTEDNETWORK_H
