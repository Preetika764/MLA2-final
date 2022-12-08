#include "Conv2D.h"

Conv2D::Conv2D(int numInputChannels, int numOutputChannels, int maskSize, int padding, int stride, int randomSeed) {
    this->stride = stride;
    this->padding = padding;
    
    std::default_random_engine generator(randomSeed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int kernelDimensions[] = {numOutputChannels, numInputChannels, maskSize, maskSize};
    masks = Vector(4, kernelDimensions);
    masks.randomInitialization(generator, distribution, sqrt(2.0 / (maskSize * maskSize * numOutputChannels)));

    int biasDimensions[] = {numOutputChannels};
    bias = Vector(1, biasDimensions);
    bias.initializeZero();
}

Vector &Conv2D::forwardPropagation(Vector &input) {
    inputImage = input;
    convolutionOutput = input.performConvolution(masks, padding, stride, bias);
    return convolutionOutput;
}

Vector Conv2D::backPropagation(Vector grads, double learningRate) {
    Vector maskGrads(masks.numDimensions, masks.dimensions);
    maskGrads.initializeZero();

    Vector inputGrads(inputImage.numDimensions, inputImage.dimensions);
    inputGrads.initializeZero();

    Vector biasGrads(1, bias.dimensions);
    biasGrads.initializeZero();

    for (int i = 0; i < inputImage.dimensions[0]; i++) {
        for (int f = 0; f < masks.dimensions[0]; f++) {
            int x = -padding;
            for (int cx = 0; cx < grads.dimensions[2]; cx++) {
                int y = -padding;
                for (int cy = 0; cy < grads.dimensions[3]; cy++) {
                    double gradValue = grads.getElementAt(i, f, cx, cy);
                    for (int fx = 0; fx < masks.dimensions[2]; fx++) {
                        int ix = x + fx;
                        if (ix >= 0 && ix < inputImage.dimensions[2]) {
                            for (int fy = 0; fy < masks.dimensions[3]; fy++) {
                                int iy = y + fy;
                                if (iy >= 0 && iy < inputImage.dimensions[3]) {
                                    for (int fc = 0; fc < masks.dimensions[1]; fc++) {
                                        maskGrads.add(f, fc, fx, fy,
                                                      inputImage.getElementAt(i, fc, ix, iy) * gradValue);
                                        inputGrads.add(i, fc, ix, iy,
                                                       masks.getElementAt(f, fc, fx, fy) * gradValue);

                                    }
                                }
                            }
                        }
                    }
                    biasGrads.add(f, gradValue);
                    y += stride;
                }
                x += stride;
            }
        }
    }
    masks -= maskGrads * learningRate;
    bias -= biasGrads * learningRate;

    return inputGrads;
}

void Conv2D::load(FILE *modelFile) {
    double value;
    for (int i = 0; i < masks.dimensions[0]; i++) {
        for (int j = 0; j < masks.dimensions[1]; j++) {
            for (int k = 0; k < masks.dimensions[2]; k++) {
                for (int l = 0; l < masks.dimensions[3]; l++) {
                    int read = fscanf(modelFile, "%lf", &value); // NOLINT(cert-err34-c)
                    if (read != 1) throw std::runtime_error("Invalid model file");
                    masks.setElementAt(i, j, k, l, value);
                }
            }
        }
    }
    for (int m = 0; m < bias.dimensions[0]; m++) {
        int read = fscanf(modelFile, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.setElementAt(m, value);
    }
}

void Conv2D::save(FILE *modelFile) {
    for (int i = 0; i < masks.dimensions[0]; i++) {
        for (int j = 0; j < masks.dimensions[1]; j++) {
            for (int k = 0; k < masks.dimensions[2]; k++) {
                for (int l = 0; l < masks.dimensions[3]; l++) {
                    fprintf(modelFile, "%.18lf ", masks.getElementAt(i, j, k, l));
                }
            }
        }
    }
    for (int m = 0; m < bias.dimensions[0]; m++) {
        fprintf(modelFile, "%.18lf ", bias.getElementAt(m));
    }
}
