#ifndef NEURAL_NET_IN_CPP_MAXPOOL_H
#define NEURAL_NET_IN_CPP_MAXPOOL_H

#include "../../AbstractNetwork/AbstractNetwork.h"
#include "../AbstractPool.h"

class MaxPool : public AbstractNetwork, public AbstractPool {
public:
    Vector inputImage;

    Vector poolingOutput;

    int poolSize;

    int stride;

    Vector targets;

    explicit MaxPool(int poolSize, int stride);

    Vector &forwardPropagation(Vector &input) override;

    Vector backPropagation(Vector grads, double learningRate) override;

    std::pair<double, int> performPooling(int inputHeightCrawler, int inputWidthCrawler, int outputHeightCrawler, int outputWidthCrawler) override;
};


#endif //NEURAL_NET_IN_CPP_MAXPOOL_H
