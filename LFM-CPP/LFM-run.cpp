#include "LFM.h"
#include <unistd.h>
#include <fstream>

int main(int argc, char* argv[]) {
    int ch;
    opterr = 0;
    string alpha_str, n_str;
    
    if(argc < 5) {
        cout << "Error: missing argument(s)."  << endl;
        cout << "\tUsage: " << argv[0] << " -n iteration_times -a alpha" << endl;
        return 1;
    }
    
    while ((ch = getopt(argc, argv, "n:a:")) != -1) {
        switch(ch) {
            case 'n': 
                n_str = string(optarg);
                break;
            case 'a':
                alpha_str = string(optarg);
                break;
        }
    }
    double alpha = stod(alpha_str);
    int n = stoi(n_str);

    ifstream intrainfile("./data/train.dat");
    
    string line, user, item;
    int rating;
    map<tuple<string, string>, int> train;


    while (intrainfile >> user >> item >> rating) {
        train.insert(make_pair(make_tuple(user, item), rating));
    }

    LFM model(n, 20, alpha, 0.8);

    time_t start, end;
    start = time(NULL);

    model.learn(train);

    end = time(NULL);

    double mse = 0;
    for (map<tuple<string, string>, int>::iterator it = train.begin(); it != train.end(); ++it) {
        auto ui = it->first;
        string user = get<0>(ui);
        string item = get<1>(ui);
        int rating = it->second;

        double rating_hat = model.predict(user, item);
        mse += (rating - rating_hat) * (rating - rating_hat);
    }

    cout << "\tMSE: " << mse / train.size() << endl;
    cout << "Learning time cost: " << end - start << "s." << endl;

    ifstream intestfile("./data/test.dat");
    ofstream outfile("./data/result.dat");
    map<string, string> predict;

    while (intestfile >> user >> item) {
        double prediction = model.predict(user, item);
        outfile << user << " " << item << " " << prediction << endl;
    }

    return 0;
}
