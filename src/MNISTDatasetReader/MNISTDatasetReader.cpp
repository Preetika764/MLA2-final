#include "MNISTDatasetReader.h"
#include "../Vector/Vector.h"

MNISTDatasetReader::MNISTDatasetReader(std::string const &trainImagesPath, std::string const &trainLabelsPath,
                                       std::string const &testImagesPath, std::string const &testLabelsPath,
                                       unsigned int batchSize) {
    this->batchSize = batchSize;
    this->currBatchPointer = 0;
    readTrainImages(trainImagesPath);
    readTrainLabels(trainLabelsPath);
    readTestImages(testImagesPath);
    readTestLabels(testLabelsPath);
}

unsigned int MNISTDatasetReader::bytesToUInt(const char *bytes) {
    return ((unsigned char) bytes[0] << 24) | ((unsigned char) bytes[1] << 16) |
           ((unsigned char) bytes[2] << 8) | ((unsigned char) bytes[3] << 0);
}

void MNISTDatasetReader::readTrainImages(std::string const &trainImagesPath) {
    std::ifstream file(trainImagesPath, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
        exit(1);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    totalTrainImages = bytesToUInt(bytes);
    file.read(bytes, 4);
    imageHeight = bytesToUInt(bytes);
    file.read(bytes, 4);
    imageWidth = bytesToUInt(bytes);

    trainImages.resize(totalTrainImages);

    char byte;
    for (int i = 0; i < totalTrainImages; ++i) {
        trainImages[i].resize(imageHeight);
        for (int j = 0; j < imageHeight; ++j) {
            trainImages[i][j].resize(imageWidth);
            for (int k = 0; k < imageWidth; ++k) {
                file.read(&byte, 1);
                trainImages[i][j][k] = (unsigned char) (byte & 0xff);
            }
        }
    }
}

void MNISTDatasetReader::readTrainLabels(std::string const &trainLabelsPath) {
    std::ifstream file(trainLabelsPath, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    totalTrainImages = bytesToUInt(bytes);

    trainLabels.resize(totalTrainImages);
    char byte;
    for (int i = 0; i < totalTrainImages; ++i) {
        file.read(&byte, 1);
        trainLabels[i] = (byte & 0xff);
    }
}

int MNISTDatasetReader::getNumTrainBatches() const {
    return (int) ((totalTrainImages % batchSize == 0) ? totalTrainImages / batchSize : (totalTrainImages / batchSize) + 1);
}

std::pair<Vector, std::vector<int>> MNISTDatasetReader::getTrainBatch() {
    int effectiveBatchSize = (int) ((totalTrainImages - currBatchPointer < batchSize) ? totalTrainImages - currBatchPointer : batchSize);

    int batchDimensions[] = {effectiveBatchSize, 1, (int) imageHeight, (int) imageWidth};
    Vector batch(4, batchDimensions);

    std::vector<int> labels;
    for (int i = 0; i < effectiveBatchSize; ++i) {
        for (int j = 0; j < imageHeight; ++j) {
            for (int k = 0; k < imageWidth; ++k) {
                batch.setElementAt(i, 0, j, k, (trainImages[currBatchPointer + i][j][k]) / 255.00);
            }
        }
        labels.push_back(trainLabels[currBatchPointer + i]);
    }

    currBatchPointer += effectiveBatchSize;
    if (currBatchPointer == totalTrainImages) {
        currBatchPointer = 0;
    }

    return std::make_pair(batch, labels);
}

void MNISTDatasetReader::readTestImages(std::string const &testImagesPath) {
    std::ifstream file(testImagesPath, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
        exit(1);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    totalTestImages = bytesToUInt(bytes);
    file.read(bytes, 4);
    imageHeight = bytesToUInt(bytes);
    file.read(bytes, 4);
    imageWidth = bytesToUInt(bytes);

    testImages.resize(totalTestImages);

    char byte;
    for (int i = 0; i < totalTestImages; ++i) {
        testImages[i].resize(imageHeight);
        for (int j = 0; j < imageHeight; ++j) {
            testImages[i][j].resize(imageWidth);
            for (int k = 0; k < imageWidth; ++k) {
                file.read(&byte, 1);
                testImages[i][j][k] = (unsigned char) (byte & 0xff);
            }
        }
    }
}

int MNISTDatasetReader::getNumTestBatches() const {
    return (int) ((totalTestImages % batchSize == 0) ? totalTestImages / batchSize : (totalTestImages / batchSize) + 1);
}

void MNISTDatasetReader::readTestLabels(std::string const &trainLabelsPath) {
    std::ifstream file(trainLabelsPath, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    totalTestImages = bytesToUInt(bytes);

    testLabels.resize(totalTestImages);
    char byte;
    for (int i = 0; i < totalTestImages; ++i) {
        file.read(&byte, 1);
        testLabels[i] = (byte & 0xff);
    }
}

std::pair<Vector, std::vector<int>> MNISTDatasetReader::getTestBatch() {
    int effectiveBatchSize = (int) ((totalTestImages - currBatchPointer < batchSize) ? totalTestImages - currBatchPointer : batchSize);

    int batchDimensions[] = {effectiveBatchSize, 1, (int) imageHeight, (int) imageWidth};
    Vector batch(4, batchDimensions);

    std::vector<int> labels;
    for (int i = 0; i < effectiveBatchSize; ++i) {
        for (int j = 0; j < imageHeight; ++j) {
            for (int k = 0; k < imageWidth; ++k) {
                batch.setElementAt(i, 0, j, k, ((double) (testImages[currBatchPointer + i][j][k])) / 255.00);
            }
        }
        labels.push_back(testLabels[currBatchPointer + i]);
    }

    currBatchPointer += effectiveBatchSize;
    if (currBatchPointer == totalTestImages) {
        currBatchPointer = 0;
    }

    return std::make_pair(batch, labels);
}
