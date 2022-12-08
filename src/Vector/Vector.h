
#ifndef NEURAL_NET_IN_CPP_Vector_H
#define NEURAL_NET_IN_CPP_Vector_H

#include <vector>
#include <random>
#include <assert.h>
#include <memory>
#include <iostream>

class Vector {
    public:
        double* elements;

        int numElements = -1;

        int numDimensions = 0;

        int dimensions[4]{};

        Vector() = default;

        Vector(int numDimensions, int const *dimensions);

        void resize(int numDimensions, int *dimensions);

        void initializeZero();

        double getElementAt(int i);

        double getElementAt(int i, int j);

        double getElementAt(int i, int j, int k);

        double getElementAt(int i, int j, int k, int l);

        void setElementAt(int i, double value);

        void setElementAt(int i, int j, double value);

        void setElementAt(int i, int j, int k, double value);

        void setElementAt(int i, int j, int k, int l, double value);

        void add(int i, double value);

        void add(int i, int j, int k, int l, double value);

        Vector matrixMultiplication(Vector B);

        Vector performConvolution(Vector masks, int padding, int stride, Vector bias);

        Vector getTranspose();

        Vector relu();

        void dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p);

        Vector getSoftmaxProbabilities();

        double sum();

        Vector reluPrime();

        Vector operator+(Vector &other);

        Vector operator*(Vector other);

        Vector operator*(double multiplier);

        Vector operator/(double divisor);

        Vector operator-=(Vector difference);

        Vector getColumnSums();

        void randomInitialization(std::default_random_engine generator, std::normal_distribution<double> distribution, double multiplier);

        void print();

        Vector &operator=(const Vector &other);

        Vector(const Vector &B);

        virtual ~Vector();

};

#endif //NEURAL_NET_IN_CPP_Vector_H
