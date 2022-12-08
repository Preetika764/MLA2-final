cmake_minimum_required(VERSION 3.5)
project(neural_net_in_cpp)

set(CMAKE_CXX_STANDARD 11)

add_executable(
        src/ANN/ANN.cpp
    src/Vector/Vector.cpp
    src/FullyConnectedNetwork/FullyConnectedNetwork.cpp
    src/SoftmaxClassifier/SoftmaxClassifier.cpp
    src/ReLU/ReLU.cpp
    src/Conv2D/Conv2D.cpp
    src/Pools/MaxPool/MaxPool.cpp
    src/Pools/MinPool/MinPool.cpp
    src/MNISTDatasetReader/MNISTDatasetReader.cpp
    src/MNISTClassifier/MNISTClassifier.cpp
)
