#ifndef CNNWITHCPP_LOADABLE_H
#define CNNWITHCPP_LOADABLE_H

#include <cstdio>

class Loadable {
    public:
        virtual void load(FILE *modelFile) = 0;

        virtual void save(FILE *modelFile) = 0;
};

#endif //CNNWITHCPP_LOADABLE_H
