#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

class LFM {
private:
    int n;          // 迭代次数
    int F;          // 隐向量维度
    double alpha;   // 学习率
    double lambda;  // 正则项参数
    double mu;      // 全局评分平均值
    map<string, vector<double>> p;  // 用户隐向量
    map<string, vector<double>> q;  // 物品隐向量
    map<string, double> bu;         // 用户偏置向量
    map<string, double> bi;         // 物品偏置向量

public:
    LFM(int n, int F, double alpha, double lambda);

    void init(map<tuple<string, string>, int> train);

    double predict(string user, string item);

    void learn(map<tuple<string, string>, int> train);

};
