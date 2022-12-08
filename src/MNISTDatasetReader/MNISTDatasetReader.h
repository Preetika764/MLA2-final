#ifndef NEURAL_NET_IN_CPP_MNISTDATASETREADER_H
#define NEURAL_NET_IN_CPP_MNISTDATASETREADER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <string.h>
#include "../Vector/Vector.h"

class MNISTDatasetReader {
public:
    unsigned int totalTrainImages = 0;

    std::vector<std::vector<std::vector<double> > > trainImages;

    std::vector<int> trainLabels;

    unsigned int totalTestImages = 0;

    std::vector<std::vector<std::vector<double> > > testImages;

    std::vector<int> testLabels;

    unsigned int currBatchPointer = 0;

    unsigned int batchSize;

    unsigned int imageHeight = 28;    // default : for MNIST

    unsigned int imageWidth = 28;    // default : for MNIST

    MNISTDatasetReader(std::string const &trainImagesPath, std::string const &trainLabelsPath,
                       std::string const &testImagesPath, std::string const &testLabelsPath, unsigned int batchSize);

    static unsigned int bytesToUInt(const char *bytes);

    void readTrainLabels(std::string const &trainLabelsPath);

    void readTrainImages(std::string const &trainImagesPath);

    void readTestLabels(std::string const &testLabelsPath);

    void readTestImages(std::string const &testImagesPath);

    int getNumTrainBatches() const;

    std::pair<Vector, std::vector<int>> getTrainBatch();

    int getNumTestBatches() const;

    std::pair<Vector, std::vector<int>> getTestBatch();
};

#endif //NEURAL_NET_IN_CPP_MNISTDATASETREADER_H