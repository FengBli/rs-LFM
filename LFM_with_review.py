# -*- coding: utf-8 -*-
"""
已考虑：
    1. 某个训练条目中没有评论数据时，更新q时则跳过，但是更新p时则对q_total随机赋值
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
import matplotlib as plt
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer


BR = '\n'
PREFIX = "./data/final-"
DEBUG = True


class LFMWithReview(object):
    """ 将评论数据考虑在内的LFM模型 """
    def __init__(self, F=20, n=2000, alpha=0.008, lambda1=0.8, lambda2=0.4):
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
        self.lambda2 = lambda2


    def __init(self, train, reviews):
        """ 初始化用户隐向量，词隐向量，以及偏置向量
        @param:
            train:      DataFrame, 训练数据集
            reviews:    dict, 每个物品对应评论词集
        新增属性:
            p:          dict, 每个用户对应隐向量
            q:          dict, 每个词对应隐向量
            bu:         dict, 每个用户对应偏置量
            bi:         dict, 每个物品对应偏置量
            mu:         float, 评分全局平均值
         """

        self.__get_dictionary(reviews)

        self.p = dict()
        self.q = dict()
        self.bu = dict()
        self.bi = dict()

        # init q
        start_time = time.time()
        fq = PREFIX + "q.pkl"
        if os.path.exists(fq):
            with open(fq, "r") as f:
                self.q = cPickle.load(f)
        else:
            for word in self.dictionary:
                self.q[word] = [random.random() / math.sqrt(self.F) for f in range(self.F)]
            with open(fq, "wb") as f:
                cPickle.dump(self.q, f)
        end_time = time.time()
        duration = end_time - start_time
        if DEBUG:
            print("dictionary word vectors initialized.total time: {}s(about {}h)".format(duration, duration_h))

        # init p, bu and bi
        start_time = time.time()
        fpb = PREFIX + "p-bu-bi.pkl"
        if os.path.exists(fpb):
            with open(fpb, "r") as f:
                self.p, self.bu, self.bi = cPickle.load(f)
        else:
            for index in range(train.count()[0]):
                u, i = train.values[index][:2]
                # init bu and bi
                self.bu[u] = 0
                self.bi[i] = 0

                # init p
                if u not in self.p:
                    self.p[u] = [random.random() / math.sqrt(self.F) for f in range(self.F)]
            # dump params
            param = [self.p, self.bu, self.bi]
            with open(fpb, "wb") as f:
                cPickle.dump(param, f)
        end_time = time.time()
        duration = end_time - start_time
        duration_h = duration / 3600
        if DEBUG:
            print("bias vectors and user vectors initialized. total time: {}s(about {}h)".format(duration, duration_h))


        # global average rating
        self.mu = round(train["Rating"].sum()*1.0 / train["Rating"].count(), 5)
        if DEBUG:
            print("global average rating: {}".format(self.mu))

    def __get_dictionary(self, reviews):
        """ 获取评论词典，并保存对象文件以备后用
        @param:
            reviews:    dict, 每个物品对应评论词集
        新增属性:
            dictionary: set, 评论字典
        """

        self.dictionary = set()

        fname =  PREFIX + "dictionary.pkl"
        if not os.path.exists(fname):
            for item in reviews:
                review = reviews[item]
                self.dictionary.update(review)

            with open(fname, mode="wb") as f:
                cPickle.dump(self.dictionary, f)
        else:
            with open(fname, mode="rb") as f:
                self.dictionary = cPickle.load(f)
        if DEBUG:
            print("dictionary loaded / dumped.")


    def predict(self, u, i, words):
        """预测用户u对物品i的评分
        @param:
            u:          string, 用户id
            i:          string, 物品id
            words:      set, 物品i对应评论词集
        @return:
            prediction: 用户u对物品i的评分预测值
        """
        prediction = self.mu + self.bu[u] + self.bi[i]

        q_total = [0] * self.F
        for word in words:
            q_total = [i + j / math.sqrt(len(words)) for i, j in zip(q_total, self.q[word])]

        for f in range(self.F):
            prediction += self.p[u][f] * q_total[f]

        return prediction


    def learn(self, train, reviews):
        """训练模型
        @param:
            train:      DataFrame, 训练数据集
            reviews:    dict, 每个物品对应评论词集
        @return:
            p:          dict, 每个用户对应隐向量
            q:          dict, 每个词对应隐向量
            bu:         dict, 每个用户对应偏置量
            bi:         dict, 每个物品对应偏置量
        """
        start_time = time.time()

        self.__init(train, reviews)

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print("initialization time: {}s".format(duration))
        total_time = 0
        for it in range(self.n):
            start_time = time.time()
            for index in range(train.count()[0]):
                u, i, rui, review_words = train.values[index]

                rui_hat = self.predict(u, i, reviews[i])

                eui = rui - rui_hat

                # updating bu and bi
                self.bu[u] += self.alpha * (eui - self.lambda2 * self.bu[u])
                self.bi[i] += self.alpha * (eui - self.lambda2 * self.bi[i])

                # updating p_u
                review_len = len(review_words)
                if review_len:
                    q_total = [0] * self.F
                    Z = math.sqrt(review_len)
                    for word in review_words:
                        q_total = [i + j for i, j in zip(q_total, self.q[word])]
                    q_total = [i / Z for i in q_total]
                else:   # 当此训练条目中无评论数据时，对q_total随机赋值
                    q_total = [random.random() for f in range(self.F)]

                for f in range(self.F):
                    self.p[u][f] += self.alpha * (q_total[f] * eui - self.lambda1 * self.p[u][f])

                # updating q_w
                if review_len:
                    for word in review_words:
                        for f in range(self.F):
                            self.q[word][f] += self.alpha * (self.p[u][f] * eui / Z - self.lambda1 * self.q[word][f])
            # end for
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            total_time += duration
            print("iteration {}, Duration: {}s".format(it, duration))
            self.alpha *= 0.9
        # end for
        print("learning total time: {}s".format(total_time))


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


def get_ratings_into_file(src_fname, dst_fname):
    """ 将训练数据的前三列，即用户id，物品id和评分提取出来，存入目标文件 """
    outf = open(dst_fname, 'a+')
    with open(src_fname, 'r') as inf:
        for line in inf.readlines():
            items = line.split(" ")[:3]
            newLine = ""
            for i in items:
                newLine += (i + " ")
            newLine = newLine.strip()
            newLine += '\n'
            outf.write(newLine)
    outf.close()


def load(fname):
    """ 将文件数据导入pandas.DataFrame。
    如果首次导入，则将此dataFrame存入同名pkl文件，方便后续调用；如果同名pkl文件存在，则直接从此pkl文件中导入。
    """
    assert os.path.exists(fname) == True, "ERROR: File not exists."
    pkl_fname = "{}.pkl".format(fname.split("/")[-1])

    if not os.path.exists(pkl_fname):
        df = pd.read_csv(fname, sep="\s+", header=None)
        df.columns = ["User", "Item", "Rating"]

        with open(pkl_fname, 'wb') as pklf:
            cPickle.dump(df, pklf)
    else:
        with open(pkl_fname, 'r') as pklf:
            df = cPickle.load(pklf)

    return df


def load_reviews(fname):
    """ 将所有评论数据导入，为每个物品构建评论词集，不考虑词频和词序，
    使用nltk库去除停用词和标点符号，并且对词进行词干化处理
    @param:
        fname:          string, 训练数据集文件名
    @return:
        train:          DataFrame, 训练数据集
        reviews:        dict, 每个物品对应的评论词集
    """

    print("loading training data")
    punctuations = {',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*'} # 自定义英文表单符号集合
    english_stopwords = set(stopwords.words("english"))
    tokenizer = WordPunctTokenizer()
    stemmer = PorterStemmer()      # 词干化处理

    reviews = dict()
    review_list = []
    fuir = PREFIX + "u-i-r.dat"
    if not os.path.exists(fuir):
        save_uir = True
        dstf = codecs.open(fuir, mode="a+", encoding="utf-8")
    else:
        save_uir = False

    with codecs.open(fname, mode='r', encoding='utf-8') as f:
        line = f.readline()
        line_no = 0
        while line:
            metas = line.split(" ")

            # 将所有(usr, item, rating)条目存为单独文件，以备基础LFM使用
            if save_uir:
                new_line = ""
                for i in metas[:3]:
                    new_line += (i + " ")
                new_line = new_line.strip()
                dstf.write(new_line + BR)

            item = metas[1]    # item id
            review_str = ""
            for word in metas[4:]:
                review_str += word + " "
            wordset = {stemmer.stem(word) for word in tokenizer.tokenize(review_str)} \
                         - punctuations - english_stopwords

            if wordset:
                review_list.append(wordset)
            else:
                review_list.append(set())

            if item not in reviews:
                reviews[item] = wordset
            else:
                reviews[item].update(wordset)

            line_no += 1
            if line_no % 1000 == 0:
                # print(".", end="")
                print("{} entries hanled.".format(line_no))
            line = f.readline()
        # end while
    # end with
    if save_uir:
        dstf.close()

    train = load(fuir)
    train["Review"] = review_list

    with open(PREFIX + "train.pkl", 'wb') as ftrain:
        cPickle.dump(train, ftrain)

    with open(PREFIX + "review.pkl", "wb") as freview:
        cPickle.dump(reviews, freview)

    return train, reviews


def main():
    ftrain = "./data/train.dat"
    train, reviews = load_reviews(ftrain)

    # with open(PREFIX + "train.pkl", "r") as ftrain:
    #     train = cPickle.load(ftrain)

    # with open(PREFIX + "review.pkl", "r") as freview:
    #     reviews = cPickle.load(freview)

    fmodel = PREFIX + "model.pkl"
    model = LFMWithReview(n=200)
    model.learn(train, reviews)
    model.dump(fmodel)
    print("======learning finished.=======")

    model2 = LFMWithReview()
    model2.load(fmodel)
    # print("u\ti\tr\tr_hat")
    print("="*50)

    count = 0
    # total = train.count()[0]
    total = 10000
    for index in range(total):
        u, i, r, words = train.values[index]
        prediction = int(round(model2.predict(u, i, words)))
        # print("{}\t{}\t{}\t{}".format(u, i, r, prediction))
        if r == prediction:
            count += 1
    print("precision = {}/{} = {}".format(count, total, count*1.0/total))



if __name__ == '__main__':
    main()
