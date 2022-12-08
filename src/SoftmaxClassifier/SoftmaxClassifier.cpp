#include "SoftmaxClassifier.h"

Vector SoftmaxClassifier::predictOutputs(Vector input) {
    output = input.getSoftmaxProbabilities();
    return output;
}

std::pair<double, Vector> SoftmaxClassifier::backPropagation(std::vector<int> groundTruths) {
    double loss = crossEntropyLoss(output, groundTruths);
    Vector gradient = crossEntropyDerivative(output, groundTruths);
    return std::make_pair(loss, gradient);
}

double SoftmaxClassifier::crossEntropyLoss(Vector &output, std::vector<int> &groundTruths) {
    double batchLoss = 0;
    int batchSize = groundTruths.size();
    for (int i = 0; i < groundTruths.size(); ++i) {
        double x = output.getElementAt(i, groundTruths[i]);
        batchLoss += -log(std::max(0.0000000001, x));
    }
    return batchLoss / batchSize;
}

Vector SoftmaxClassifier::crossEntropyDerivative(Vector &output, std::vector<int> &groundTruths) {
    Vector result = output;
    for (int i = 0; i < groundTruths.size(); ++i) {
        result.setElementAt(i, groundTruths[i], result.getElementAt(i, groundTruths[i]) - 1);
    }
    return result / output.dimensions[0];
}