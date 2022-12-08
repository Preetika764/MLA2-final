#ifndef NEURAL_NET_IN_CPP_ANN_H
#define NEURAL_NET_IN_CPP_ANN_H

#include <vector>
#include "../Vector/Vector.h"
#include "../AbstractNetwork/AbstractNetwork.h"
#include "../SoftmaxClassifier/SoftmaxClassifier.h"

class ANN {
    public:
        int numNetworks;

        AbstractNetwork** networks;

        SoftmaxClassifier* classifier;

        double learningRate;

        ANN(int numNetworks, AbstractNetwork **networks, SoftmaxClassifier *classifier, double learningRate);

        double trainBatch(Vector &input, std::vector<int> &groundTruths);

        Vector feedForward(Vector &input);

        std::vector<int> predictOutputs(Vector &input);

        std::vector<double> predictOutputs1(Vector &input);

        std::vector<double> predictOutputs2(Vector &input);

        std::vector<double> predictOutputs3(Vector &input);

        void load(std::string path);

        void save(std::string path);
};

#endif //NEURAL_NET_IN_CPP_ANN_H
