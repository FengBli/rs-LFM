#-*- coding: utf-8 -*-

"""
latent factor model without considering reviews.
"""

from __future__ import  print_function
import codecs
import cPickle
import math
import random
import os
import time
import numpy as np
import pandas as pd


PREFIX = "test-"
DEBUG = True


class LFM(object):
    """ 将评论数据考虑在内的LFM模型 """
    def __init__(self, F=20, n=2000, alpha=0.008, lambda1=0.8):
        """@param:
            F:          隐向量维度
            n:          模型SGD迭代次数
            alpha:      SGD学习率
            lambda1:    用户、物品正则项参数
            lambda2:    偏置项正则项参数
        """
        self.F = F
        self.n = n
        self.alpha = alpha
        self.lambda1 = lambda1


    def __init(self, train):
        """ 初始化带偏置量的LFM模型里的偏置参数和隐变量参数 """
        self.p = dict()
        self.q = dict()
        self.bu = dict()
        self.bi = dict()

        for index in range(train.count()[0]):
            u, i, rui = list(train.values[index])
            self.bu[u] = 0
            self.bi[i] = 0
            if u not in self.p:
                self.p[u] = [random.random() / math.sqrt(self.F) for f in range(self.F)]
            if i not in self.q:
                self.q[i] = [random.random() / math.sqrt(self.F) for f in range(self.F)]
        # end for

        self.mu = round(train["Rating"].sum()*1.0 / train["Rating"].count(), 5)


    def predict(self, u, i):
        """ 运用带偏置量的模型进行预测 """
        prediction = self.mu + self.bu[u] + self.bi[i]

        for f in range(self.F):
            prediction += self.p[u][f] * self.q[i][f]

        return prediction


    def learn(self, train):
        """ 学习带有偏置量的LFM模型 """
        self.__init(train)
        for step in range(self.n):

            start_time = time.time()

            for index in range(train.count()[0]):
                u, i, rui = train.values[index]
                rui_hat = self.predict(u, i)
                eui = rui - rui_hat

                # updating parameters
                self.bu[u] += self.alpha * (eui - self.lambda1 * self.bu[u])
                self.bi[i] += self.alpha * (eui - self.lambda1 * self.bi[i])

                for f in range(self.F):
                    self.p[u][f] += self.alpha * (self.q[i][f] * eui - self.lambda1 * self.p[u][f])
                    self.q[i][f] += self.alpha * (self.p[u][f] * eui - self.lambda1 * self.q[i][f])
            # end for

            end_time = time.time()
            duration = end_time - start_time
            print("Iteration {}, time cost: {}s".format(step, duration))

            self.alpha *= 0.9
        # end for


    def dump(self, fname):
        """ 保存模型，以备后用，保存相关参数：p, q, bu, bi
        @param:
            fname:      string, 模型保存的文件路径名
        """
        # assert os.path.exists(fname) == False, "ERROR: file already exists"

        param = [self.p, self.q, self.bu, self.bi, self.mu]
        with open(fname, "wb") as f:
            cPickle.dump(param, f)
        if DEBUG:
            print("model dumped.")


    def load(self, fname):
        """ 从对象文件中导入模型 """
        with open(fname, "rb") as f:
            self.p, self.q, self.bu, self.bi, self.mu = cPickle.load(f)

        if DEBUG:
            print("model loaded.")


def main():
    fmodel = PREFIX + "model.pkl"
    # ftrain = "final-1-train.pkl"
    ftrain = "small-train.pkl"
    with open(ftrain, "rb") as f:
        train = cPickle.load(f)
    print("Train data loaded.")


    model = LFM(n=200)
    model.learn(train)
    print("="*50)

    model.dump(fmodel)


if __name__ == '__main__':
    main()
