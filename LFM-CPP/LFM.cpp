/* LFM with consideration of bias */
#include "LFM.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>


LFM::LFM(int n, int F, double alpha, double lambda) {
    this->n = n;
    this->F = F;
    this->alpha = alpha;
    this->lambda = lambda;
    this->mu = 0.0;
}

void LFM::init(map<tuple<string, string>, int> train) {

    srand(time(NULL));

    for(map<tuple<string, string>, int>::iterator it = train.begin(); it != train.end(); ++it) {
        auto ui = it->first;
        string user = get<0>(ui);
        string item = get<1>(ui);

        int rating = it->second;
        this->mu += rating;

        // initialize bias vetors
        this->bu[user] = 0.0;
        this->bi[item] = 0.0;

        map<string, vector<double>>::iterator userIt, itemIt;

        // initialize user vectors
        userIt = this->p.find(user);
        if(userIt == this->p.end()) {
            vector<double> v;
            for (int i = 0; i < this->F; i++){
                vector<double>::iterator vit = v.end();
                double value = (rand() % 1000 + 1.0) / 1000 / sqrt(this->F);
                v.insert(vit, value);
            }
            this->p[user] = v;
        }

        // initialize item vectors
        itemIt = this->q.find(item);
        if(itemIt == this->q.end()) {
            vector<double> v;
            for (int i = 0; i < this->F; i++) {
                vector<double>::iterator vit = v.end();
                double value = (rand() % 1000 + 1.0) / 1000 / sqrt(this->F);
                v.insert(vit, value);
            }
            this->q[item] = v;
        }

    }
    this->mu /= train.size();
}

double LFM::predict(string user, string item) {
    double prediction = this->mu + this->bu[user] + this->bi[item];
    for(int f = 0; f < this->F; ++f)
        prediction += this->p[user][f] * this->q[item][f];
    return prediction;
}

void LFM::learn(map<tuple<string, string>, int> train) {
    this->init(train);
    time_t start, end;
    double mse;
    for (int step = 0; step < this->n; ++step) {
        start = time(NULL);
        for(map<tuple<string, string>, int>::iterator it = train.begin(); it != train.end(); ++it) {
            auto ui = it->first;
            string user = get<0>(ui);
            string item = get<1>(ui);
            int rating = it->second;

            double rating_hat = this->predict(user, item);
            double eui = rating - rating_hat;
            mse += pow(eui, 2);

            // updating bias vectors
            this->bu[user] += this->alpha * (eui - this->lambda * this->bu[user]);
            this->bi[item] += this->alpha * (eui - this->lambda * this->bi[item]);

            // updating user and item vectors
            for (int f = 0; f < this->F; ++f) {
                this->p[user][f] += this->alpha * (this->q[item][f] * eui - this->lambda * this->p[user][f]);
                this->q[item][f] += this->alpha * (this->p[user][f] * eui - this->lambda * this->q[item][f]);
            }
        }
        end = time(NULL);
        mse /= train.size();
        cout << "\tMSE: " << mse << "\t" << endl;
        cout << "Iteration " << step << ", time cost: " << end - start << "s,";
        this->alpha *= 0.9;
    }
}
