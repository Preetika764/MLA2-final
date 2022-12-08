#ifndef NEURAL_NET_IN_CPP_MINPOOL_H
#define NEURAL_NET_IN_CPP_MINPOOL_H

#include "../../AbstractNetwork/AbstractNetwork.h"
#include "../AbstractPool.h"

class MinPool : public AbstractNetwork, public AbstractPool {
    public:
        explicit MinPool(int poolSize, int stride);

        Vector &forwardPropagation(Vector &input) override;

        std::pair<double, int> performPooling(int inputHeightCrawler, int inputWidthCrawler, int outputHeightCrawler,
                                              int outputWidthCrawler) override;

        Vector backPropagation(Vector grads, double learningRate) override;
};


#endif //NEURAL_NET_IN_CPP_MINPOOL_H
