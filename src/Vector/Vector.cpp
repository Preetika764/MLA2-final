#include "Vector.h"
#include <cstring>

Vector::Vector(int numDimensions, int const *dimensions) {
    assert(numDimensions > 0 && numDimensions <= 4);
    this->numDimensions = numDimensions;

    int size = 1;
    for (int i = 0; i < numDimensions; i++) {
        size *= dimensions[i];
        this->dimensions[i] = dimensions[i];
    }
    numElements = size;
    elements = new double[numElements];
}

void Vector::initializeZero() {
    memset(elements, 0, sizeof(double) * numElements);
}

void Vector::randomInitialization(std::default_random_engine generator, std::normal_distribution<double> distribution,
                                  double multiplier) {
    for (int i = 0; i < numElements; i++) {
        elements[i] = distribution(generator) * multiplier;
    }
}

double Vector::getElementAt(int i) {
    assert(numDimensions == 1);
    return elements[i];
}

double Vector::getElementAt(int i, int j) {
    assert(numDimensions == 2);
    return elements[j + i * dimensions[1]];
}

double Vector::getElementAt(int i, int j, int k) {
    assert(numDimensions == 3);
    return elements[k + j * dimensions[2] + i * dimensions[1] * dimensions[2]];
}

double Vector::getElementAt(int i, int j, int k, int l) {
    assert(numDimensions == 4);
    return elements[l + k * dimensions[3] + j * dimensions[2] * dimensions[3] + i * dimensions[1] * dimensions[2] * dimensions[3]];
}


void Vector::setElementAt(int i, double value) {
    elements[i] = value;
}

void Vector::setElementAt(int i, int j, double value) {
    assert(numDimensions == 2);
    elements[j + i * dimensions[1]] = value;
}

void Vector::setElementAt(int i, int j, int k, double value) {
    assert(numDimensions == 3);
    elements[k + j * dimensions[2] + i * dimensions[1] * dimensions[2]] = value;
}

void Vector::setElementAt(int i, int j, int k, int l, double value) {
    assert(numDimensions == 4);
    elements[l + k * dimensions[3] + j * dimensions[2] * dimensions[3] + i * dimensions[1] * dimensions[2] * dimensions[3]] = value;
}

void Vector::add(int i, double value) {
    elements[i] += value;
}

void Vector::add(int i, int j, int k, int l, double value) {
    assert(numDimensions == 4);
    elements[l + k * dimensions[3] + j * dimensions[2] * dimensions[3] + i * dimensions[1] * dimensions[2] * dimensions[3]] += value;
}

void Vector::resize(int numDimensions, int *dimensions) {
    assert(numDimensions > 0 && numDimensions <= 4);
    this->numDimensions = numDimensions;
    std::copy(dimensions, dimensions + 4, this->dimensions);
}

Vector::Vector(const Vector &B) : numElements(B.numElements), numDimensions(B.numDimensions),
                                  elements(new double[B.numElements]) {
    std::copy(B.elements, B.elements + numElements, elements);
    std::copy(B.dimensions, B.dimensions + 4, dimensions);
}


Vector::~Vector() {
    delete[] elements;
}

Vector Vector::matrixMultiplication(Vector B) {
    assert(numDimensions == 2 && B.numDimensions == 2);
    assert(dimensions[1] == B.dimensions[0]);

    int resultDimensions[] = {dimensions[0], B.dimensions[1]};
    Vector product(2, resultDimensions);
    for (int i = 0; i < this->dimensions[0]; i++) {
        for (int j = 0; j < B.dimensions[1]; j++) {
            double value = 0;
            for (int k = 0; k < B.dimensions[0]; k++) {
                value += this->getElementAt(i, k) * B.getElementAt(k, j);
            }
            product.setElementAt(i, j, value);
        }
    }
    return product;
}


Vector Vector::getTranspose() {
    assert(numDimensions == 2);
    int new_dims[] = {dimensions[1], dimensions[0]};
    Vector transpose(numDimensions, new_dims);
    for (int i = 0; i < dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            transpose.setElementAt(j, i, getElementAt(i, j));
        }
    }

    return transpose;
}

Vector Vector::performConvolution(Vector masks, int padding, int stride, Vector bias) {
    assert(masks.dimensions[1] == dimensions[1]);

    int outputHeight = (dimensions[2] + 2 * padding - masks.dimensions[2]) / stride + 1;
    int outputWidth = (dimensions[3] + 2 * padding - masks.dimensions[3]) / stride + 1;
    int outputDimensions[] = {dimensions[0], masks.dimensions[0], outputHeight, outputWidth};
    Vector output(4, outputDimensions);

    for (int i = 0; i < dimensions[0]; i++) {
        for (int j = 0; j < masks.dimensions[0]; j++) {
            for (int k = 0; k < outputHeight; k++) {
                for (int l = 0; l < outputWidth; l++) {
                    int scrollingStartIndexX = k * stride - padding;
                    int scrollingStartIndexY = l * stride - padding;

                    double convValue = 0;
                    for (int m = 0; m < masks.dimensions[1]; m++) {
                        for (int n = 0; n < masks.dimensions[2]; n++) {
                            for (int o = 0; o < masks.dimensions[3]; o++) {
                                int x = scrollingStartIndexX + n;
                                int y = scrollingStartIndexY + o;

                                if (x >= 0 && x < dimensions[2] && y >= 0 && y < dimensions[3]) {
                                    double a = getElementAt(i, m, x, y);
                                    double b = masks.getElementAt(j, m, n, o);
                                    convValue += (a * b);
                                }
                            }
                        }
                    }
                    output.setElementAt(i, j, k, l, convValue + bias.getElementAt(j));
                }
            }
        }
    }
    return output;
}

Vector Vector::relu() {
    Vector result(numDimensions, dimensions);
    for (int i = 0; i < numElements; i++) {
        double x = elements[i];
        result.elements[i] = x > 0 ? x : 0;
    }

    return result;
}

Vector Vector::reluPrime() {
    Vector prime(numDimensions, dimensions);
    for (int i = 0; i < numElements; i++) {
        prime.elements[i] = elements[i] > 0 ? 1 : 0;
    }
    return prime;
}

double Vector::sum() {
    double total = 0;
    for (int i = 0; i < numElements; i++) {
        total += elements[i];
    }
    return 0;
}

Vector Vector::getSoftmaxProbabilities() {
    assert(numDimensions == 2);

    Vector result(2, dimensions);
    for (int i = 0; i < dimensions[0]; i++) {
        double maxInDimension = -1;
        for (int j = 0; j < dimensions[1]; j++) {
            if (j == 0 || getElementAt(i, j) > maxInDimension) {
                maxInDimension = getElementAt(i, j);
            }
        }

        double sumExponentiatedDeviations = 0;
        for (int j = 0; j < dimensions[1]; j++) {
            double x = getElementAt(i, j);
            sumExponentiatedDeviations += exp(getElementAt(i, j) - maxInDimension);
        }

        for (int j = 0; j < dimensions[1]; j++) {
            result.setElementAt(i, j, exp(getElementAt(i, j) - maxInDimension) / sumExponentiatedDeviations);
        }
    }
    return result;
}

Vector Vector::operator+(Vector &other) {
    Vector result(numDimensions, dimensions);
    if (other.numDimensions == 1 && other.numElements == dimensions[1] && numDimensions == 2) {
        for (int k = 0; k < dimensions[0]; k++) {
            for (int j = 0; j < dimensions[1]; j++) {
                result.setElementAt(k, j, getElementAt(k, j) + other.getElementAt(j));
            }
        }
    } else if (other.numDimensions == numDimensions && other.numElements == numElements) {
        for (int i = 0; i < numElements; i++) {
            result.elements[i] = elements[i] + other.elements[i];
        }
    }
    return result;
    throw std::logic_error("Undefined result");
}


Vector Vector::operator*(Vector other) {
    assert(numElements == other.numElements);
    Vector product(numDimensions, dimensions);
    for (int i = 0; i < numElements; i++) {
        product.elements[i] = elements[i] * other.elements[i];
    }
    return product;
}


Vector Vector::operator*(double multiplier) {
    Vector product(numDimensions, dimensions);
    for (int i = 0; i < numElements; i++) {
        product.elements[i] = elements[i] * multiplier;
    }
    return product;
}

Vector Vector::operator/(double divisor) {
    Vector quotient(numDimensions, dimensions);
    for (int i = 0; i < numElements; i++) {
        quotient.elements[i] = elements[i] / divisor;
    }
    return quotient;
}

Vector Vector::operator-=(Vector difference) {
    assert(numElements == difference.numElements);
    for (int i = 0; i < numElements; i++) {
        elements[i] = elements[i] - difference.elements[i];
    }
    return *this;
}

Vector Vector::getColumnSums() {
    assert(numDimensions == 2);

    int resultDimensions[] = {dimensions[1]};
    Vector result(1, resultDimensions);
    for (int j = 0; j < dimensions[1]; j++) {
        double colSum = 0;
        for (int i = 0; i < dimensions[0]; i++) {
            colSum += getElementAt(i, j);
        }
        result.setElementAt(j, colSum);
    }
    return result;
}

void Vector::print() {
    if (numDimensions == 2) {
        int rows = dimensions[0], cols = dimensions[1];
        std::cout << "Vector2D (" << rows << ", " << cols << ")\n[";
        for (int i = 0; i < rows; i++) {
            if (i != 0) std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < cols; j++) {
                if (j == (cols - 1)) {
                    printf("%.18lf", getElementAt(i, j));
                } else {
                    printf("%.18lf ", getElementAt(i, j));
                }

            }
            if (i == (rows - 1)) {
                std::cout << "]]\n";
            } else {
                std::cout << "]\n";
            }
        }
    } else {
        printf("Vector%dd (", numDimensions);
        for (int i = 0; i < numDimensions; i++) {
            printf("%d", dimensions[i]);
            if (i != (numDimensions - 1)) {
                printf(",");
            }
        }
        printf(")\n[");
        for (int j = 0; j < numElements; j++) {
            printf("%lf ", elements[j]);
        }
        printf("]\n");
    }
}

Vector &Vector::operator=(const Vector &other) {
    if (this != &other) {
        double *new_data = new double[other.numElements];
        std::copy(other.elements, other.elements + other.numElements, new_data);
        if (numElements != -1) {
            delete[] elements;
        }
        numElements = other.numElements;
        std::copy(other.dimensions, other.dimensions + 4, dimensions);
        numDimensions = other.numDimensions;
        elements = new_data;
    }

    return *this;
}

void Vector::dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p) {
    for (int i = 0; i < numElements; i++) {
        elements[i] = (distribution(generator) < p) / p;
    }
}
