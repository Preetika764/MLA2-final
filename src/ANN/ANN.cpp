#include "ANN.h"
#include "../Loadable/Loadable.h"
#include <bits/stdc++.h>

using namespace std;

ANN::ANN(int numNetworks, AbstractNetwork **networks, SoftmaxClassifier *classifier, double learningRate) {
    this->numNetworks = numNetworks;
    this->networks = networks;
    this->classifier = classifier;
    this->learningRate = learningRate;
}

double ANN::trainBatch(Vector &input, vector<int>& groundTruths) {
    Vector output = feedForward(input);
    pair<double, Vector> backPropagationResult = classifier->backPropagation(groundTruths);
    double batchLoss = backPropagationResult.first;
    Vector grads = backPropagationResult.second;
    for (int i = numNetworks - 1; i >= 0; --i) {
        grads = networks[i]->backPropagation(grads, 0.2);
    }
    return batchLoss;
}

Vector ANN::feedForward(Vector &input) {
    for (int i = 0; i < numNetworks; i++) {
        input = networks[i]->forwardPropagation(input);
    }
    return classifier->predictOutputs(input);
}

std::vector<int> ANN::predictOutputs(Vector &input) {
    Vector output = feedForward(input);
    std::vector<int> predictions(output.dimensions[0]);
    for (int i = 0; i < output.dimensions[0]; ++i) {
        int predictionIndex = 0;
        double maxProbability = -1;
        for (int j = 0; j < output.dimensions[1]; ++j) {
            if (output.getElementAt(i, j) > maxProbability) {
                maxProbability = output.getElementAt(i, j);
                predictionIndex = j;
            }
        }
        predictions[i] = predictionIndex;
    }
    return predictions;
}

std::vector<double> ANN::predictOutputs1(Vector &input) {
    Vector output = feedForward(input);
    std::vector<double> predictions(output.dimensions[0]);
    for (int i = 0; i < output.dimensions[0]; ++i) {
        int predictionIndex = 0;
        double maxProbability = -1;
        for (int j = 0; j < output.dimensions[1]; ++j) {
            if (output.getElementAt(i, j) > maxProbability) {
                maxProbability = output.getElementAt(i, j);
                predictionIndex = j;
            }
        }
        predictions[i] = predictionIndex;
    }
    return predictions;
}

std::vector<double> ANN::predictOutputs2(Vector &input) {
    Vector output = feedForward(input);
    std::vector<double> predictions(output.dimensions[0]);
    for (int i = 0; i < output.dimensions[0]; ++i) {
        double sumProbability = 0.00;
        for (int j = 0; j < output.dimensions[1]; ++j) {
            double pr = output.getElementAt(i, j);
            sumProbability += pr*log10(pr);    
        }
        predictions[i] = sumProbability;
        // cout << sumProbability << "\n";
    }
    return predictions;
}

std::vector<double> ANN::predictOutputs3(Vector &input) {
    Vector output = feedForward(input);
    std::vector<double> predictions(output.dimensions[0]);
    for (int i = 0; i < output.dimensions[0]; ++i) {
        double maxProbability = -1;
        double secondMaxProbability = -1;
        vector<double> temp;
        for (int j = 0; j < output.dimensions[1]; ++j) {
            {
                temp.push_back(output.getElementAt(i,j));
            }
            sort(temp.begin(), temp.end(), greater<double>());
        }
        predictions.push_back(temp[0]-temp[1]);
        // cout << temp[0] - temp[1] << "\n";
        }
    return predictions;
}

void ANN::load(std::string path) {
    FILE *model_file = fopen(path.c_str(), "r");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (int i = 0; i < numNetworks; i++) {
        if(Loadable* network = dynamic_cast<Loadable*>(networks[i])){
            network->load(model_file);
        }
    }
}

void ANN::save(std::string path) {
    FILE *model_file = fopen(path.c_str(), "w");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (int i = 0; i < numNetworks; i++) {
        if(Loadable* network = dynamic_cast<Loadable*>(networks[i])){
            network->save(model_file);
        }
    }
}
