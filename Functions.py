# -*- coding: utf-8 -*-

"""
Functions definitions for MP2


"""

import numpy as np
import re
import nltk
from scipy import sparse
from scipy.sparse import vstack, hstack
from sklearn.metrics import accuracy_score
import xgboost as xgb

# %%  Bernoulli Naive Bayes from scratch 

def get_w_bernouliiNB(thetas_o):
    w0 = np.log(1-thetas_o)
    w1 = np.log(thetas_o)

    return w0, w1


def get_theta_bernouliiNB(X_bi):

    n_fea = X_bi.shape[1]
    # X_add = np.concatenate((X_bi, np.ones((1, n_fea)), np.zeros((1, n_fea))), axis=0)
    X_add = vstack((X_bi, sparse.csr_matrix(np.ones((1, n_fea))), sparse.csr_matrix(np.zeros((1, n_fea)))))
    thetas = np.array(X_add.mean(axis=0)).reshape((-1,))

    return thetas


class BinaryCntVec(object):

    def __init__(self, max_df=1.0, min_df=0.0):
        self.vocab_dict = None
        self.vocab = None
        self.docu_size = None
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, tokenized_list):
        self.vocab_dict = dict()
        self.vocab = list()
        self.docu_size = len(tokenized_list)
        for data_point in tokenized_list:
            temp = list(set(data_point)) # remove repetitive words
            for word in temp:
                if word in self.vocab_dict:
                    self.vocab_dict[word] += 1
                else:
                    self.vocab_dict[word] = 1
        self.vocab = [word for word in self.vocab_dict if self.min_df * self.docu_size <= self.vocab_dict[word] <= self.max_df * self.docu_size]
        return self

    def transform(self, tokenized_list):
        mat = list()
        for word in self.vocab:
            col = [1 if word in set(data_point) else 0 for data_point in tokenized_list]
            mat.append(col)

        return np.array(mat).T

    def fit_transform(self, tokenized_list):
        return self.fit(tokenized_list).transform(tokenized_list)


class bernoulli_NB(object):

    def __init__(self):
        self.dictionary = dict()
        self.w = None
        self.w_bias = None

    def fit(self, X, y):
        n_fea = X.shape[1]
        n_class = len(set(list(y)))
        self.w = np.empty((0, n_fea))
        self.w_bias = np.empty((0,))
        for label in range(n_class):
            oneclass_index = [k for k in range(y.size) if y[k] == label]
            oneclass_P = float(len(oneclass_index)) / y.size
            X_o = X[oneclass_index, :]
            # X_r = np.delete(X.toarray(), oneclass_index, axis=0)
            # print(X_o.shape, X_r.shape)

            theta_o = get_theta_bernouliiNB(X_o)
            # theta_r = get_theta_bernouliiNB(X_r)

            w0, w1 = get_w_bernouliiNB(theta_o)

            self.w = np.append(self.w, (w1 - w0).reshape((1, -1)), axis=0)
            self.w_bias = np.append(self.w_bias, (w0.sum() + np.log(oneclass_P)))

        return self

    def predict(self, X):
        res = X.dot(self.w.T) + self.w_bias
        return np.argmax(res, axis=1)


# %%  other functions


class_label = {'hockey': 0, 'nba': 1, 'leagueoflegends': 2, 'soccer': 3, 'funny': 4, 'movies': 5, 'anime': 6,
               'Overwatch': 7, 'trees': 8, 'GlobalOffensive': 9, 'nfl': 10, 'AskReddit': 11, 'gameofthrones': 12,
               'conspiracy': 13, 'worldnews': 14, 'wow': 15, 'europe': 16, 'canada': 17, 'Music': 18, 'baseball': 19}


# rough split the classes into 5 meta class:
# 0 - sports 1 - games 2 - others 3 - entertainment 4 - world
metaclass_label = {0: 0,
                   1: 0,
                   2: 1,
                   3: 0,
                   4: 2,
                   5: 3,
                   6: 3,
                   7: 1,
                   8: 2,
                   9: 1,
                   10: 0,
                   11: 2,
                   12: 3,
                   13: 4,
                   14: 4,
                   15: 1,
                   16: 4,
                   17: 4,
                   18: 3,
                   19: 0}


def itself(obj):
    return obj


def raw_text_process(text,*punc_args):
    for punc in punc_args:
        text = text.replace(punc,'')
    return text.lower().split()


def snowball_stem(text):

    pro_words = list()
    sbws = nltk.stem.snowball.SnowballStemmer('english', ignore_stopwords=True)

    for word in nltk.word_tokenize(text):
        pro_words.append(sbws.stem(word))

    return pro_words


def porter_stem(text):

    pro_words = list()
    ptws = nltk.stem.PorterStemmer()

    for word in nltk.word_tokenize(text):
        pro_words.append(ptws.stem(word))

    return pro_words


def lemmatize(text, tokenized=False):

    pro_words = list()
    wnl = nltk.WordNetLemmatizer()

    for word, tag in nltk.pos_tag(text if tokenized else nltk.word_tokenize(text)):
    # for word, tag in nltk.pos_tag(raw_text_process(text)):

        if(word == 'brightest'):
            pass

        if tag.startswith('NN'):
            ori_word = wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            ori_word = wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            ori_word = wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            ori_word = wnl.lemmatize(word, pos='r')
        else:
            ori_word = word

        pro_words.append(ori_word)

    return pro_words


def cleandata(raw_data):

    temp = raw_data.lower()

    # temp = re.sub('<.*?>', '', temp)
    # temp = re.sub('^\s+|\s+', ' ', temp)
    # temp = re.sub(r'^https?:\/\/.*[\r\n]*', '', temp, flags=re.MULTILINE)
    # # temp = re.sub('\d+', '', temp)
    # temp = re.sub('\'s', ' ', temp)
    # temp = re.sub('\'ve ', ' have', temp)
    # temp = re.sub('\'d ', ' would', temp)
    # temp = re.sub('\'ll', ' will', temp)
    # temp = re.sub('can\'t', 'cannot', temp)
    # temp = re.sub('can not', 'cannot', temp)
    temp = re.sub('[^a-zA-Z0-9]', ' ', temp)

    cleantext = temp

    return cleantext


def text_preprocess(text, n_lemm=0, stem=None):
    if stem == 'Snowball':
        return snowball_stem(text)
    elif stem == 'Porter':
        return porter_stem(text)
    elif n_lemm == 0:
        return nltk.word_tokenize(cleandata(text))
    elif n_lemm == 1:
        return lemmatize(cleandata(text))
    elif n_lemm == 2:
        return lemmatize(lemmatize(cleandata(text)), tokenized=True)
    else:
        return None


class expand_feature(object):

    def __init__(self):
        self.norm = None

    def fit_transform(self, X, token_list):
        sl_temp = list()
        for item in token_list:
            sl_temp.append(len(item))
        X_sent_length = np.array(sl_temp)
        self.norm = X_sent_length.mean(axis=0)
        X_sl_norm = X_sent_length.reshape(-1, 1) / self.norm
        X = sparse.hstack([X, X_sl_norm])
        X = sparse.csr_matrix(X)
        return X

    def transform(self, X, token_list):
        sl_temp = list()
        for item in token_list:
            sl_temp.append(len(item))
        X_sent_length = np.array(sl_temp)
        X_sl_norm = X_sent_length.reshape(-1, 1) / self.norm
        X = sparse.hstack([X, X_sl_norm])
        X = sparse.csr_matrix(X)
        return X


def feature_subset_test(X, y, train_index, val_index, classifier, selector, n_list):

    result = dict()

    for item in n_list:
        selector.k = item
        X_new = selector.fit_transform(X, y)

        trainx, valx = X_new[train_index, :], X_new[val_index, :]  # corresponding rows
        trainy, valy = y[train_index], y[val_index]

        classifier.fit(trainx, trainy)
        y_predict = classifier.predict(valx)

        result[item] = accuracy_score(valy, y_predict)
        print(accuracy_score(valy, y_predict))

    return result


def get_subindex(Y_meta, meta_n):
    return [k for k in range(Y_meta.size) if Y_meta[k] == meta_n]


def stage1_feature_trans(y_st1, n_class):
    n_data = y_st1.shape[0]
    n_model = y_st1.shape[1]
    binary_features = np.empty([n_data, 0])
    for row in y_st1.T:
        # row_model = np.empty([0, n_class])
        row_model = np.zeros((n_data, n_class))
        for label, i in zip(row, range(n_data)):
            # temp = np.zeros([1, n_class])
            row_model[i, int(label)] = 1
            # row_model = np.append(row_model, temp, axis=0)

        # row_model = np.delete(row_model, -1, axis=1)
        binary_features = np.append(binary_features, row_model, axis=1)

    return binary_features


class Fold4CVStacking(object):

    def __init__(self, clfs, trainx, trainy):
        self.clfs = clfs
        self.xgbparam = None
        self.trainx = trainx
        self.trainy = trainy

    def fit_train(self, n_fold, n_stage, mode='hard', reprocess=False):

        trainx = self.trainx
        trainy = self.trainy

        train_size = int(trainy.size / 4)
        trainx1, trainy1 = trainx[:train_size], trainy[:train_size]
        trainx2, trainy2 = trainx[train_size:2*train_size], trainy[train_size:2*train_size]
        trainx3, trainy3 = trainx[2*train_size:3*train_size], trainy[2*train_size:3*train_size]
        trainx4, trainy4 = trainx[3*train_size:], trainy[3*train_size:]

        # print(trainx1.shape, trainx2.shape, trainx3.shape, trainx4.shape)

        if reprocess:
            y_st_hard = list()
            y_st_soft = np.empty((trainy.size, 0))
            for clf in self.clfs:

                hard_flag = False

                try:
                    trainxR = vstack((trainx2, trainx3, trainx4))
                except ValueError:
                    trainxR = np.vstack((trainx2, trainx3, trainx4))
                trainyR = np.concatenate((trainy2, trainy3, trainy4))
                clf.fit(trainxR, np.ravel(trainyR))

                if mode == 'soft':
                    try:
                        y_pred1 = clf.predict_proba(trainx1)
                    except AttributeError:
                        hard_flag = True
                        y_pred1 = clf.predict(trainx1)
                else:
                    y_pred1 = clf.predict(trainx1)


                try:
                    trainxR = vstack((trainx1, trainx3, trainx4))
                except ValueError:
                    trainxR = np.vstack((trainx1, trainx3, trainx4))
                trainyR = np.concatenate((trainy1, trainy3, trainy4))
                clf.fit(trainxR, np.ravel(trainyR))
                if mode == 'soft':
                    try:
                        y_pred2 = clf.predict_proba(trainx2)
                    except AttributeError:
                        y_pred2 = clf.predict(trainx2)
                else:
                    y_pred2 = clf.predict(trainx2)

                try:
                    trainxR = vstack((trainx1, trainx2, trainx4))
                except ValueError:
                    trainxR = np.vstack((trainx1, trainx2, trainx4))
                trainyR = np.concatenate((trainy1, trainy2, trainy4))
                clf.fit(trainxR, np.ravel(trainyR))
                if mode == 'soft':
                    try:
                        y_pred3 = clf.predict_proba(trainx3)
                    except AttributeError:
                        y_pred3 = clf.predict(trainx3)
                else:
                    y_pred3 = clf.predict(trainx3)

                try:
                    trainxR = vstack((trainx1, trainx2, trainx3))
                except ValueError:
                    trainxR = np.vstack((trainx1, trainx2, trainx3))
                trainyR = np.concatenate((trainy1, trainy2, trainy3))
                clf.fit(trainxR, np.ravel(trainyR))
                if mode == 'soft':
                    try:
                        y_pred4 = clf.predict_proba(trainx4)
                    except AttributeError:
                        y_pred4 = clf.predict(trainx4)
                else:
                    y_pred4 = clf.predict(trainx4)

                if hard_flag:
                    y_pred = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4))
                    y_st_hard.append(list(y_pred))
                else:
                    y_pred = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4), axis=0)
                    if mode == 'soft':
                        y_st_soft = np.concatenate((y_st_soft, y_pred), axis=1)
                    else:
                        y_st_hard.append(list(y_pred))

            y_st_hard = np.array(y_st_hard).T
            y_st_hard = stage1_feature_trans(y_st_hard, n_class=len(set(list(trainy))))

            st_feature = np.hstack((y_st_hard, y_st_soft))
            save_train = st_feature
            np.save('st' + str(n_stage) + '_train' + str(n_fold), save_train)
        else:
            st_feature = np.load('st' + str(n_stage) + '_train' + str(n_fold) + '.npy')

        return st_feature

    def predict_val(self, valx, n_fold, n_stage, mode='hard', reprocess=False):
        if reprocess:
            y_st_hard = list()
            y_st_soft = np.empty((valx.shape[0], 0))
            for clf in self.clfs:
                hard_flag = False

                clf.fit(self.trainx, np.ravel(self.trainy))
                if mode == 'soft':
                    try:
                        y_pred = clf.predict_proba(valx)
                    except AttributeError:
                        hard_flag = True
                        y_pred = clf.predict(valx)
                else:
                    y_pred = clf.predict(valx)

                if hard_flag:
                    y_st_hard.append(list(y_pred))
                else:
                    if mode == 'soft':
                        y_st_soft = np.concatenate((y_st_soft, y_pred), axis=1)
                    else:
                        y_st_hard.append(list(y_pred))

            y_st_hard = np.array(y_st_hard).T
            y_st_hard = stage1_feature_trans(y_st_hard, n_class=len(set(list(self.trainy))))

            st_feature_val = np.hstack((y_st_hard, y_st_soft))
            save_val = st_feature_val
            np.save('st' + str(n_stage) + '_val' + str(n_fold), save_val)
        else:
            st_feature_val = np.load('st' + str(n_stage) + '_val' + str(n_fold) + '.npy')

        return st_feature_val

    def fit_train_xgb(self, params, n_fold, n_stage, reprocess=False):

        self.xgbparam = params

        trainx = self.trainx
        trainy = self.trainy

        train_size = int(trainy.size / 4)
        trainx1, trainy1 = trainx[:train_size], trainy[:train_size]
        trainx2, trainy2 = trainx[train_size:2 * train_size], trainy[train_size:2 * train_size]
        trainx3, trainy3 = trainx[2 * train_size:3 * train_size], trainy[2 * train_size:3 * train_size]
        trainx4, trainy4 = trainx[3 * train_size:], trainy[3 * train_size:]

        if reprocess:

            try:
                trainxR = vstack((trainx2, trainx3, trainx4))
            except ValueError:
                trainxR = np.vstack((trainx2, trainx3, trainx4))
            trainyR = np.concatenate((trainy2, trainy3, trainy4))
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred1 = model.predict(xgb.DMatrix(trainx1))

            try:
                trainxR = vstack((trainx1, trainx3, trainx4))
            except ValueError:
                trainxR = np.vstack((trainx1, trainx3, trainx4))
            trainyR = np.concatenate((trainy1, trainy3, trainy4))
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred2 = model.predict(xgb.DMatrix(trainx2))

            try:
                trainxR = vstack((trainx1, trainx2, trainx4))
            except ValueError:
                trainxR = np.vstack((trainx1, trainx2, trainx4))
            trainyR = np.concatenate((trainy1, trainy2, trainy4))
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred3 = model.predict(xgb.DMatrix(trainx3))

            try:
                trainxR = vstack((trainx1, trainx2, trainx3))
            except ValueError:
                trainxR = np.vstack((trainx1, trainx2, trainx3))
            trainyR = np.concatenate((trainy1, trainy2, trainy3))
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred4 = model.predict(xgb.DMatrix(trainx4))

            y_pred = np.concatenate((y_pred1, y_pred2, y_pred3, y_pred4), axis=0)

            if params['objective'] == 'multi:softmax':
                y_pred = stage1_feature_trans(y_pred.reshape((-1, 1)), n_class=len(set(list(self.trainy))))

            save_train = y_pred
            np.save('st' + str(n_stage) + '_xgb_train' + str(n_fold), save_train)
        else:
            y_pred = np.load('st' + str(n_stage) + '_xgb_train' + str(n_fold) + '.npy')

        return y_pred

    def predict_val_xgb(self, valx, n_fold, n_stage, reprocess=False):
        if reprocess:
            model = xgb.train(self.xgbparam, xgb.DMatrix(self.trainx, self.trainy))
            y_pred = model.predict(xgb.DMatrix(valx))

            if self.xgbparam['objective'] == 'multi:softmax':
                y_pred = stage1_feature_trans(y_pred.reshape((-1, 1)), n_class=len(set(list(self.trainy))))

            save_val = y_pred
            np.save('st' + str(n_stage) + '_xgb_val' + str(n_fold), save_val)
        else:
            y_pred = np.load('st' + str(n_stage) + '_xgb_val' + str(n_fold) + '.npy')

        return y_pred


class Fold2CVStacking(object):

    def __init__(self, clfs, trainx, trainy):
        self.clfs = clfs
        self.xgbparam = None
        self.trainx = trainx
        self.trainy = trainy

    def fit_train(self, n_fold, n_stage, mode='hard', reprocess=False):

        trainx = self.trainx
        trainy = self.trainy

        train_size = int(trainy.size / 2)
        trainx1, trainy1 = trainx[:train_size], trainy[:train_size]
        trainx2, trainy2 = trainx[train_size:], trainy[train_size:]

        if reprocess:
            y_st_hard = list()
            y_st_soft = np.empty((trainy.size, 0))
            for clf in self.clfs:

                hard_flag = False

                trainxR = trainx2
                trainyR = trainy2
                clf.fit(trainxR, np.ravel(trainyR))

                if mode == 'soft':
                    try:
                        y_pred1 = clf.predict_proba(trainx1)
                    except AttributeError:
                        hard_flag = True
                        y_pred1 = clf.predict(trainx1)
                else:
                    y_pred1 = clf.predict(trainx1)

                trainxR = trainx1
                trainyR = trainy1
                clf.fit(trainxR, np.ravel(trainyR))
                if mode == 'soft':
                    try:
                        y_pred2 = clf.predict_proba(trainx2)
                    except AttributeError:
                        y_pred2 = clf.predict(trainx2)
                else:
                    y_pred2 = clf.predict(trainx2)

                if hard_flag:
                    y_pred = np.concatenate((y_pred1, y_pred2))
                    y_st_hard.append(list(y_pred))
                else:
                    y_pred = np.concatenate((y_pred1, y_pred2), axis=0)
                    if mode == 'soft':
                        y_st_soft = np.concatenate((y_st_soft, y_pred), axis=1)
                    else:
                        y_st_hard.append(list(y_pred))

            y_st_hard = np.array(y_st_hard).T
            y_st_hard = stage1_feature_trans(y_st_hard, n_class=len(set(list(trainy))))

            st_feature = np.hstack((y_st_hard, y_st_soft))
            save_train = st_feature
            np.save('st' + str(n_stage) + '_train' + str(n_fold), save_train)
        else:
            st_feature = np.load('st' + str(n_stage) + '_train' + str(n_fold) + '.npy')

        return st_feature

    def predict_val(self, valx, n_fold, n_stage, mode='hard', reprocess=False):
        if reprocess:
            y_st_hard = list()
            y_st_soft = np.empty((valx.shape[0], 0))
            for clf in self.clfs:
                hard_flag = False

                clf.fit(self.trainx, np.ravel(self.trainy))
                if mode == 'soft':
                    try:
                        y_pred = clf.predict_proba(valx)
                    except AttributeError:
                        hard_flag = True
                        y_pred = clf.predict(valx)
                else:
                    y_pred = clf.predict(valx)

                if hard_flag:
                    y_st_hard.append(list(y_pred))
                else:
                    if mode == 'soft':
                        y_st_soft = np.concatenate((y_st_soft, y_pred), axis=1)
                    else:
                        y_st_hard.append(list(y_pred))

            y_st_hard = np.array(y_st_hard).T
            y_st_hard = stage1_feature_trans(y_st_hard, n_class=len(set(list(self.trainy))))

            st_feature_val = np.hstack((y_st_hard, y_st_soft))
            save_val = st_feature_val
            np.save('st' + str(n_stage) + '_val' + str(n_fold), save_val)
        else:
            st_feature_val = np.load('st' + str(n_stage) + '_val' + str(n_fold) + '.npy')

        return st_feature_val

    def fit_train_xgb(self, params, n_fold, n_stage, reprocess=False):

        self.xgbparam = params

        trainx = self.trainx
        trainy = self.trainy

        train_size = int(trainy.size / 2)
        trainx1, trainy1 = trainx[:train_size], trainy[:train_size]
        trainx2, trainy2 = trainx[train_size:], trainy[train_size:]

        if reprocess:

            trainxR = trainx2
            trainyR = trainy2
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred1 = model.predict(xgb.DMatrix(trainx1))

            trainxR = trainx1
            trainyR = trainy1
            model = xgb.train(params, xgb.DMatrix(trainxR, trainyR))
            y_pred2 = model.predict(xgb.DMatrix(trainx2))

            y_pred = np.concatenate((y_pred1, y_pred2), axis=0)

            if params['objective'] == 'multi:softmax':
                y_pred = stage1_feature_trans(y_pred.reshape((-1, 1)), n_class=len(set(list(self.trainy))))

            save_train = y_pred
            np.save('st' + str(n_stage) + '_xgb_train' + str(n_fold), save_train)
        else:
            y_pred = np.load('st' + str(n_stage) + '_xgb_train' + str(n_fold) + '.npy')

        return y_pred

    def predict_val_xgb(self, valx, n_fold, n_stage, reprocess=False):
        if reprocess:
            model = xgb.train(self.xgbparam, xgb.DMatrix(self.trainx, self.trainy))
            y_pred = model.predict(xgb.DMatrix(valx))

            if self.xgbparam['objective'] == 'multi:softmax':
                y_pred = stage1_feature_trans(y_pred.reshape((-1, 1)), n_class=len(set(list(self.trainy))))

            save_val = y_pred
            np.save('st' + str(n_stage) + '_xgb_val' + str(n_fold), save_val)
        else:
            y_pred = np.load('st' + str(n_stage) + '_xgb_val' + str(n_fold) + '.npy')

        return y_pred


def empty_column_detect(mat):
    count = 0
    for column in mat.T:
        if np.count_nonzero(column.toarray()) == 0:
            count = count + 1

    return count

















