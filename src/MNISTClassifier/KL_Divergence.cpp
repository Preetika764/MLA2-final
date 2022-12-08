#include <iostream>
#include <fstream>
#include <bits/stdc++.h>

#include "../ANN/ANN.h"
#include "../FullyConnectedNetwork/FullyConnectedNetwork.h"
#include "../MNISTDatasetReader/MNISTDatasetReader.h"
#include "../ReLU/ReLU.h"
#include "../Conv2D/Conv2D.h"
#include "../Pools/MaxPool/MaxPool.h"
#include "../Constants/MNISTConstants.h"

using namespace std;

void writeBatchLossToFile(int numBatches, double* losses){
    ofstream csvfile;
    csvfile.open("/tmp/losses.txt");

    for(int i = 0; i < numBatches; i++){
        csvfile << i+1 << "," << losses[i] << "\n";
    }
    csvfile.close();
}

MNISTDatasetReader loadDataset(){
    const std::string trainImagesPath = DATASET_PATH + MNIST_TRAIN_IMAGES_PATH;
    const std::string trainLabelsPath = DATASET_PATH + MNIST_TRAIN_LABELS_PATH;
    const std::string testImagesPath = DATASET_PATH + MNIST_TEST_IMAGES_PATH;
    const std::string testLabelsPath = DATASET_PATH + MNIST_TEST_LABELS_PATH;
    MNISTDatasetReader MNISTDataset(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath, BATCH_SIZE);
    return MNISTDataset;
}

void trainingProcess(MNISTDatasetReader &dataset, ANN &model){
    int numTrainBatches = dataset.getNumTrainBatches();

    double losses[NUM_EPOCHS*numTrainBatches];
    for (int k = 0; k < NUM_EPOCHS; ++k) {
        // cout << "* ============================== TRAINING EPOCH " << k + 1 << " ==============================\n";
        for (int i = 0; i < numTrainBatches; ++i) {
            pair<Vector, vector<int> > trainBatch = dataset.getTrainBatch();
            Vector images = trainBatch.first;
            vector<int> labels = trainBatch.second;
            double loss = model.trainBatch(images, labels);
            if ((i + 1) % 10 == 0) {
                //cout << "\r* Batch " << i+1 << "/" << numTrainBatches << ": Batch loss " << loss;
                fflush(stdout);
            }
            losses[NUM_EPOCHS*k + i] = loss;
        }
    }
    model.save(MODEL_FILE_PATH);

    writeBatchLossToFile(NUM_EPOCHS*numTrainBatches, losses);
}

double testingProcess(MNISTDatasetReader dataset, ANN model){
    // cout << "* ================================= TESTING ====================================\n";
    int correctPredictions = 0;
    int totalTestcases = 0;
    int numTestBatches = dataset.getNumTestBatches();
    for (int i = 0; i < numTestBatches; ++i) {
        if ((i + 1) % 10 == 0 || i == (numTestBatches - 1)) {
            // cout << "\r* Batch " <<  i + 1 << "/" << numTestBatches << ": testing complete...";
            fflush(stdout);
        }
        pair<Vector, vector<int> > testBatch = dataset.getTestBatch();
        vector<int> outputs = model.predictOutputs(testBatch.first);
        for (int j = 0; j < outputs.size(); ++j) {
            if (outputs[j] == testBatch.second[j]) {
                correctPredictions++;
            }
        }
        totalTestcases += (int) testBatch.second.size();
    }

    return  ((double) correctPredictions * 100) / totalTestcases;
}

double testingProcessBagging(MNISTDatasetReader dataset, ANN *model1, ANN *model2, ANN *model3, ANN *model4, ANN *model5){
    cout << "* ================================= TESTING ====================================\n";
    int correctPredictions = 0;
    int totalTestcases = 0;
    // cout << "Testing\n";
    int numTestBatches = dataset.getNumTestBatches();
    // cout << "Let's see\n";
    cout << numTestBatches << "\n";
    for (int i = 0; i < numTestBatches; ++i) {
        if ((i + 1) % 10 == 0 || i == (numTestBatches - 1)) {
            cout << "\r* Batch " <<  i + 1 << "/" << numTestBatches << ": testing complete...";
            //fflush(stdout);
        }
        pair<Vector, vector<int> > testBatch1 = dataset.getTestBatch();
        pair<Vector, vector<int> > testBatch2 = testBatch1;
        pair<Vector, vector<int> > testBatch3 = testBatch1;
        pair<Vector, vector<int> > testBatch4 = testBatch1;
        pair<Vector, vector<int> > testBatch5 = testBatch1; 
        // cout << "Check\n";
        vector<int> outputs1 = (*model1).predictOutputs(testBatch1.first);
        vector<int> outputs2 = (*model2).predictOutputs(testBatch2.first);
        vector<int> outputs3 = (*model3).predictOutputs(testBatch3.first);
        vector<int> outputs4 = (*model4).predictOutputs(testBatch4.first);
        vector<int> outputs5 = (*model5).predictOutputs(testBatch5.first);
        vector<int> outputs;
        // outputs has majority vote
        for (int j = 0; j < outputs1.size(); ++j) {
            int count[10] = {0};
            count[outputs1[j]]++;
            count[outputs2[j]]++;
            count[outputs3[j]]++;
            count[outputs4[j]]++;
            count[outputs5[j]]++;
            int max = 0;
            int maxIndex = 0;
            for (int k = 0; k < 10; ++k) {
                if (count[k] > max) {
                    max = count[k];
                    maxIndex = k;
                }
            }
            outputs.push_back(maxIndex);
        }
        for (int j = 0; j < outputs.size(); ++j) {
            if (outputs[j] == testBatch1.second[j]) {
                correctPredictions++;
            }
        }
        totalTestcases += (int) testBatch1.second.size();
    }

    return  ((double) correctPredictions * 100) / totalTestcases;
}

pair<vector<std::vector<std::vector<double>>>,vector<int>> baggingSample(vector<std::vector<std::vector<double>>> images,vector<int> labels)
{
    vector<std::vector<std::vector<double>>> baggingImages;
    vector<int> baggingLabels;
    for (int i = 0; i < images.size(); i++)
    {
        int random = rand() % images.size();
        baggingImages.push_back(images[random]);
        baggingLabels.push_back(labels[random]);
    }
    return make_pair(baggingImages,baggingLabels);
}

vector<double> KLDivergence(vector<int> * prediction1, vector<int> * prediction2,vector<int> * prediction3, vector<int> * prediction4, vector<int> * prediction5)
{
    vector<double> entropies;
    for (int i = 0; i < (*prediction1).size(); i++)
    {
        int numbers[10] = {0};
        double y = 0.00;
        numbers[(*prediction1)[i]]++;
        numbers[(*prediction2)[i]]++;
        numbers[(*prediction3)[i]]++;
        numbers[(*prediction4)[i]]++;
        numbers[(*prediction5)[i]]++;
        for (int j = 0; j < 10; j++)
        {
            if(numbers[j] == 0)
                continue;
            else    
                y += (numbers[j]/5) * log10(numbers[j]/5);
        }
        entropies.push_back(y);
    }
    return entropies;
}

void specialSort(std::vector<std::vector<std::vector<double> > > * unlabeledPool, std::vector<int> * unlabeledPoolLabels, vector<double> * entropies){

    vector<std::pair<double, std::pair<std::vector<std::vector<double>>, int>>> v;
    for (int i = 0; i < (*unlabeledPoolLabels).size(); i++)
    {
        auto p = make_pair((*unlabeledPool)[i], (*unlabeledPoolLabels)[i]);
        auto q = make_pair((*entropies)[i], p);
        v.push_back(q);
    }

    sort(v.begin(), v.end());
    (*unlabeledPool).clear();
    (*unlabeledPoolLabels).clear();
    (*entropies).clear();
    for (int j = 0; j < v.size(); j++)
    {
        auto p = v[j].first;
        auto q = v[j].second.first;
        auto r = v[j].second.second;
        (*unlabeledPool).push_back(q);
        (*unlabeledPoolLabels).push_back(r);
        (*entropies).push_back(p);
    }
}

/*
int main(int argc, char** argv) {
    if(argc == 2){
        APP_ROOT = (string) argv[1];
    }

    double accuracy;
    double labels;
    MNISTDatasetReader dataset = loadDataset();
    AbstractNetwork* modules[] = {
        new Conv2D(1, 8, 3, 0, 1, RANDOM_SEED),
        new MaxPool(2, 2),
        new ReLU(),
        new FullyConnectedNetwork(1352, 30, RANDOM_SEED),
        new ReLU(),
        new FullyConnectedNetwork(30, 10, RANDOM_SEED)
    };
    ANN model = ANN(6, modules, new SoftmaxClassifier(), LEARNING_RATE);

    std::vector<std::vector<std::vector<double> > > labeledPool((int)dataset.totalTrainImages * 0.1);

    std::vector<int> labeledPoolLabels((int)dataset.totalTrainImages * 0.1);

    std::vector<std::vector<std::vector<double> > > unlabeledPool((int)dataset.totalTrainImages * 0.9);

    std::vector<int> unlabeledPoolLabels ((int)dataset.totalTrainImages * 0.9);
    
    int starting = 0, ending = (int)dataset.totalTrainImages * 0.1;
    copy(dataset.trainImages.begin() + starting, dataset.trainImages.begin() + ending, labeledPool.begin());
    copy(dataset.trainLabels.begin() + starting, dataset.trainLabels.begin() + ending, labeledPoolLabels.begin());
    copy(dataset.trainImages.begin() + ending, dataset.trainImages.end(), unlabeledPool.begin());
    copy(dataset.trainLabels.begin() + ending, dataset.trainLabels.end(), unlabeledPoolLabels.begin());

    // std::cout << labeledPool.size() << "\n";
    // std::cout << unlabeledPool.size() << "\n";
    // std::cout << labeledPoolLabels.size() << "\n";
    // std::cout << unlabeledPoolLabels.size() << "\n";

    cout <<  "Initializing the active learner with the labeled pool\n";

    dataset.totalTrainImages = labeledPool.size();
    dataset.trainImages = labeledPool;
    dataset.trainLabels = labeledPoolLabels;
    dataset.currBatchPointer = 0;

    int numTrainBatches = dataset.getNumTrainBatches();

    // testing the model with initial 10% labeled pool
    cout << "Using entropy for sampling\n";
    cout << "Labels\tAccuracy\n";

    trainingProcess(dataset, model);
    accuracy = testingProcess(dataset, model);
    labels = labeledPool.size();
    cout << labels << "\t" << accuracy << "\n";

    // active learning loop
    int X = 6000;
    // X = 120; // comment or remove this line, just doing this to see if I'm correct
    for (int i = 0; i < X/600; i++)
    {
        //cout << "\nIteration no.: " << i+1 << "\n";
        // TODO : sort the unlabeled pool using uncertainty sampling
        // assuming the unlabeled pool is sorted
        
        dataset.totalTrainImages = unlabeledPool.size();
        dataset.trainImages = unlabeledPool;
        dataset.trainLabels = unlabeledPoolLabels;
        dataset.currBatchPointer = 0;
        numTrainBatches = dataset.getNumTrainBatches();

        // vector<double> softmax;
        vector<double> entropies;
        // vector<double> smallMargin;
        for (int i = 0; i < numTrainBatches; ++i) {
            pair<Vector, vector<int> > trainBatch = dataset.getTrainBatch();
            Vector images = trainBatch.first;
            vector<int> labels = trainBatch.second;
            //  vector<double> tempSoftmax = model.predictOutputs1(images);
            vector<double> tempEntropies = model.predictOutputs2(images);
            // vector<double> tempSmallMargin = model.predictOutputs3(images);
            for (int j = 0; j < tempEntropies.size(); j++)
            {
                // softmax.push_back(tempSoftmax[j]);
                 entropies.push_back(tempEntropies[j]);
            }
        }

        //cout << "Size of softmax: " << softmax.size() << "\n";

        // add the four methods for sorting
        specialSort(&unlabeledPool, &unlabeledPoolLabels, &entropies);
        // adding 60 images from unlabeled to labeled pool
        for (int j = 0; j < 600; j++)
        {
            labeledPool.push_back(unlabeledPool[j]);
            labeledPoolLabels.push_back(unlabeledPoolLabels[j]);
            unlabeledPool.erase(unlabeledPool.begin(), unlabeledPool.begin() + 1);
            unlabeledPoolLabels.erase(unlabeledPoolLabels.begin(), unlabeledPoolLabels.begin() + 1);
        }

        // reinitialize the dataset with the new labeled pool
        dataset.totalTrainImages = labeledPool.size();
        dataset.trainImages = labeledPool;
        dataset.trainLabels = labeledPoolLabels;
        dataset.currBatchPointer = 0;

        // retrain the model
        trainingProcess(dataset, model);
        accuracy = testingProcess(dataset, model);
        labels = labeledPool.size();
        cout << labels << "\t" << accuracy << "\n";
    }
    return 0;
}

*/


int main(int argc, char** argv) {
    if(argc == 2){
        APP_ROOT = (string) argv[1];
    }
    double accuracy;
    int values;
    MNISTDatasetReader dataset = loadDataset();
    AbstractNetwork* modules[] = {
        new Conv2D(1, 8, 3, 0, 1, RANDOM_SEED),
        new MaxPool(2, 2),
        new ReLU(),
        new FullyConnectedNetwork(1352, 30, RANDOM_SEED),
        new ReLU(),
        new FullyConnectedNetwork(30, 10, RANDOM_SEED)
    };
    vector<ANN> committee(5, ANN(6, modules, new SoftmaxClassifier(), LEARNING_RATE));

    cout << "Committee initalization done\n";
    std::vector<std::vector<std::vector<double> > > labeledPool((int)dataset.totalTrainImages * 0.1);
    std::vector<int> labeledPoolLabels((int)dataset.totalTrainImages * 0.1);
    std::vector<std::vector<std::vector<double> > > unlabeledPool((int)dataset.totalTrainImages * 0.9);
    std::vector<int> unlabeledPoolLabels ((int)dataset.totalTrainImages * 0.9);
    
    int starting = 0, ending = (int)dataset.totalTrainImages * 0.1;
    copy(dataset.trainImages.begin() + starting, dataset.trainImages.begin() + ending, labeledPool.begin());
    copy(dataset.trainLabels.begin() + starting, dataset.trainLabels.begin() + ending, labeledPoolLabels.begin());
    copy(dataset.trainImages.begin() + ending, dataset.trainImages.end(), unlabeledPool.begin());
    copy(dataset.trainLabels.begin() + ending, dataset.trainLabels.end(), unlabeledPoolLabels.begin());

    cout << "committee getting trained by bagging\n";
    for (int i = 0; i < 5; i++)
    {
        dataset.totalTrainImages = labeledPool.size();
        // cout << "I reach here\n";
        pair<vector<std::vector<std::vector<double>>>,vector<int>> sample = baggingSample(labeledPool, labeledPoolLabels); 
        dataset.trainImages = sample.first;
        dataset.trainLabels = sample.second;
        dataset.currBatchPointer = 0;
        // cout << "fsdg\n";
        trainingProcess(dataset, committee[i]);
    }

    // cout << values << "\t" << testingProcess(dataset, committee[0]) << "\n";
    cout << values << "\t" << testingProcessBagging(dataset, &committee[0], &committee[1], &committee[2], &committee[3], &committee[4]) << "\n";
    cout << "Initial training of committee with bagging\n";

    // active learning loop
    int X = 6000;
    // X = 60; // comment or remove this line, just doing this to see if I'm correct
    for (int i = 0; i < X/600; i++)
    {
        cout << "\nIteration no.: " << i+1 << "\n";
        // Get the predictions

        vector<int> predictions[5];
        for (int i = 0; i < 5; i++)
        {
            dataset.totalTrainImages = unlabeledPool.size();
            dataset.trainImages = unlabeledPool;
            dataset.trainLabels = unlabeledPoolLabels;
            dataset.currBatchPointer = 0;
            int numTrainBatches = dataset.getNumTrainBatches();
            for (int j = 0; j < numTrainBatches; j++)
            {
                pair<Vector, vector<int> > trainBatch = dataset.getTrainBatch();
                Vector images = trainBatch.first;
                vector<int> labels = trainBatch.second;
                vector<int> p = committee[i].predictOutputs(images);
                for (int k = 0; k < p.size(); k++)
                    predictions[i].push_back(p[k]);
            }
        }

        vector<double> voteEntropy = KLDivergence(&predictions[0], &predictions[1], &predictions[2], &predictions[3],&predictions[4]);        
        // add the two methods for sorting
        // cout << "Voting done\n";KLDivergence
        // adding 60 images from unlabeled to labeled pool
        specialSort(&unlabeledPool, &unlabeledPoolLabels, &voteEntropy);

        for (int j = 0; j < 600; j++)
        {
            labeledPool.push_back(unlabeledPool[j]);
            labeledPoolLabels.push_back(unlabeledPoolLabels[j]);
            unlabeledPool.erase(unlabeledPool.begin(), unlabeledPool.begin() + 1);
            unlabeledPoolLabels.erase(unlabeledPoolLabels.begin(), unlabeledPoolLabels.begin() + 1);
        }
        // cout << "Made new labeled and unlabeled pools\n";
        // reinitialize the dataset with the new labeled pool
        dataset.totalTrainImages = labeledPool.size();
        dataset.trainImages = labeledPool;
        dataset.trainLabels = labeledPoolLabels;
        dataset.currBatchPointer = 0;

        for (int i = 0; i < 5; i++)
        {
            dataset.totalTrainImages = labeledPool.size();
            pair<vector<std::vector<std::vector<double>>>,vector<int>> sample = baggingSample(labeledPool, labeledPoolLabels); 
            dataset.trainImages = sample.first;
            dataset.trainLabels = sample.second;
            dataset.currBatchPointer = 0;
            int numTrainBatches = dataset.getNumTrainBatches();
            trainingProcess(dataset, committee[i]);
        }

        cout << values << "\t" << testingProcessBagging(dataset, &committee[0], &committee[1], &committee[2], &committee[3], &committee[4]) << "\n";
        // "Retraining done\n";
    }
    return 0;
}
