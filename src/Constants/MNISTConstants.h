#ifndef CNNWITHCPP_CONSTANTS_H
#define CNNWITHCPP_CONSTANTS_H

#include <stdio.h>
#include <iostream>
#include <string>
#include <filesystem>

std::string APP_ROOT = "/home/preetika/MLA2";

constexpr static double LEARNING_RATE = 0.2;

constexpr static int BATCH_SIZE = 60;

constexpr static int RANDOM_SEED = 0;

static std::string DATASET_PATH = APP_ROOT + "/data";

const static std::string MNIST_TRAIN_IMAGES_PATH = "/train-images-idx3-ubyte";

const static std::string MNIST_TRAIN_LABELS_PATH = "/train-labels-idx1-ubyte";

const static std::string MNIST_TEST_IMAGES_PATH = "/t10k-images-idx3-ubyte";

const static std::string MNIST_TEST_LABELS_PATH = "/t10k-labels-idx1-ubyte";

const static int NUM_EPOCHS = 1;

const static std::string MODEL_FILE_PATH = APP_ROOT + "/src/mnist.txt";

#endif //CNNWITHCPP_CONSTANTS_H
