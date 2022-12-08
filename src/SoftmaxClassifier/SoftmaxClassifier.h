#ifndef NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
#define NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H

#include "../Vector/Vector.h"

class SoftmaxClassifier {
    public:
        Vector output;

        Vector predictOutputs(Vector input);

        std::pair<double, Vector> backPropagation(std::vector<int> groundTruths);

        double crossEntropyLoss(Vector &output, std::vector<int> &groundTruths);

        Vector crossEntropyDerivative(Vector &output, std::vector<int> &groundTruths);
};

#endif //NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
