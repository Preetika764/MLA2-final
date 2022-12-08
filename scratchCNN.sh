#! /bin/bash

echo "Compiling source files. Using C++11..."

g++ --std=c++11 src/ANN/ANN.cpp src/Vector/Vector.cpp src/FullyConnectedNetwork/FullyConnectedNetwork.cpp src/SoftmaxClassifier/SoftmaxClassifier.cpp src/ReLU/ReLU.cpp src/Conv2D/Conv2D.cpp src/Pools/MaxPool/MaxPool.cpp src/Pools/MinPool/MinPool.cpp src/MNISTDatasetReader/MNISTDatasetReader.cpp src/MNISTClassifier/UncertaintySampling.cpp -o ScratchCNN


echo "Source has been compiled. Starting application..."
echo ""
echo ""
sleep 2

if [[ "$OSTYPE" == "win32" ]]; then
  cls
else
  clear
fi

./ScratchCNN ${PWD}