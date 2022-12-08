#ifndef NEURAL_NET_IN_CPP_CONV2D_H
#define NEURAL_NET_IN_CPP_CONV2D_H

#include "../AbstractNetwork/AbstractNetwork.h"
#include "../Loadable/Loadable.h"

class Conv2D : public AbstractNetwork, public Loadable {
public:
    Vector inputImage;

    Vector convolutionOutput;

    Vector masks;

    int padding;

    int stride;

    Vector bias;

    Conv2D(int numInputChannels, int numOutputChannels, int maskSize, int padding, int stride, int randomSeed = 0);

    Vector &forwardPropagation(Vector &input) override;

    Vector backPropagation(Vector chain_gradient, double learningRate) override;

    void load(FILE *modelFile) override;

    void save(FILE *modelFile) override;
};


#endif //NEURAL_NET_IN_CPP_CONV2D_H
