#ifndef CNNWITHCPP_ABSTRACTPOOL_H
#define CNNWITHCPP_ABSTRACTPOOL_H

#include "../Vector/Vector.h"

class AbstractPool {
    public:
        Vector inputImage;

        Vector poolingOutput;

        int poolSize;

        int stride;

        Vector targets;

        virtual std::pair<double, int>
        performPooling(int inputHeightCrawler, int inputWidthCrawler, int outputHeightCrawler, int outputWidthCrawler) = 0;
};

#endif //CNNWITHCPP_ABSTRACTPOOL_H
