# -*- coding: utf-8 -*-


import Functions as f
import numpy as np
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingCVClassifier
import xgboost as xgb

import csv
import json
import time

from Functions import metaclass_label
from Functions import class_label
inverse_label = {v: k for k, v in class_label.items()}


re_preprocess = False
testing = False
re_preprocess_test = False


if __name__ == '__main__':

    file_path = 'reddit_train.csv'
    tokenized_list = list()
    y_list = list()
    n_lemm = 1
    token_path = 'Tokens_' + str(n_lemm) + '_v4.json'

    if re_preprocess:
        with open(file_path, encoding='utf-8', mode='r') as csv_file:
            data_reader = csv.reader(csv_file, delimiter=',')

            feature_name = next(data_reader)
            i = 0
            for item in data_reader:
                tokenized_list.append(f.text_preprocess(item[1], n_lemm, stem=None))
                y_list.append(class_label[item[2]])
                i = i + 1
                if i % 1000 == 0: print(i)
        with open(token_path, 'w', encoding='utf-8') as file:
            data_save = [tokenized_list, y_list]
            json.dump(data_save, file, ensure_ascii=False)
    else:
        with open(token_path, 'r', encoding='utf-8') as file:
            tokenized_list, y_list = json.load(file)

    X = tokenized_list
    Y = np.array(y_list)

    y_temp = list()
    for item in y_list:
        y_temp.append(metaclass_label[item])
    Y_meta = np.array(y_temp)

    if not testing:

        fold = 5
        kf = KFold(n_splits=fold, random_state=13839, shuffle=True)  # random state = 13839 for Group 23
        accuracy_SVM = np.zeros(fold)
        k = 0

        time_start = time.time()
        for train_index, val_index in kf.split(X):

            trainx_list, valx_list = [X[i] for i in train_index], [X[i] for i in val_index]  # corresponding rows
            trainy, valy = Y[train_index], Y[val_index]
            trainy_meta, valy_meta = Y_meta[train_index], Y_meta[val_index]

            cvc = CountVectorizer(stop_words=stopwords.words('english'), tokenizer=f.itself, preprocessor=None,
                                  lowercase=False, max_df=1.0, min_df=2,
                                  ngram_range=(1, 2), binary=True)  # 0.000014 is minimum min_df
            idftrans = TfidfTransformer()
            epf = f.expand_feature()
            selector = SelectKBest(chi2, k=30000)

            trainx = cvc.fit_transform(trainx_list)
            print(len(cvc.get_feature_names()))
            trainx = idftrans.fit_transform(trainx)
            trainx = epf.fit_transform(trainx, trainx_list)
            trainx = selector.fit_transform(trainx, trainy)

            valx = cvc.transform(valx_list)
            valx = idftrans.transform(valx)
            valx = epf.transform(valx, valx_list)
            valx = selector.transform(valx)

            # bicvc = CountVectorizer(stop_words=stopwords.words('english'), tokenizer=f.itself, preprocessor=None,
            #                         lowercase=False, max_df=1.0, min_df=0.0001,
            #                         ngram_range=(1, 2), binary=True)
            # bicvc = f.BinaryCntVec(max_df=1.0, min_df=0.001)
            # trainx = bicvc.fit_transform(trainx_list)
            # valx = bicvc.transform(valx_list)

            # clf = f.bernoulli_NB()
            # clf = BernoulliNB()
            


            # train_size = int(trainy.size / 2)
            # trainx1, trainy1 = trainx[:train_size], trainy[:train_size]
            # trainx2, trainy2 = trainx[train_size:], trainy[train_size:]


            clfs = list()
            clfs.append(svm.LinearSVC(penalty='l2', dual=False, C=0.4, multi_class='ovr', tol=1e-4))
            clfs.append(MultinomialNB(alpha=0.1))
            clfs.append(LogisticRegression(C=5, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr'))
            clfs.append(RandomForestClassifier(n_estimators=1500, max_depth=120, min_samples_leaf=1,
                                               random_state=18, max_features='log2', n_jobs=-1))

            # cvst = f.Fold4CVStacking(clfs, trainx, trainy)
            cvst = f.Fold2CVStacking(clfs, trainx, trainy)

            repro = False

            bi_feat_train = cvst.fit_train(n_fold=k, n_stage=1, reprocess=repro)
            bi_feat_val = cvst.predict_val(valx, n_fold=k, n_stage=1, reprocess=repro)

        
            bi_feat_train = f.stage1_feature_trans(bi_feat_train, n_class=20)
            bi_feat_val = f.stage1_feature_trans(bi_feat_val, n_class=20)

            feat_train = bi_feat_train
            feat_val = bi_feat_val

            params = {'booster': 'gblinear',
                      'learning_rate': 0.05,
                      'eta': 0.01,
                      'max_depth': 20,
                      'min_child_weight': 1,
                      'num_boost_round': 30,
                      'objective': 'multi:softmax',
                      'gamma': 0,
                      'lambda': 3e-6,
                      'random_state': 19,
                      'silent': 0,
                      'num_class': 20
                      }
            # xgb_feat_train = cvst.fit_train_xgb(params, n_fold=k, n_stage=1, reprocess=repro)
            # xgb_feat_val = cvst.predict_val_xgb(valx, n_fold=k, n_stage=1, reprocess=repro)
            #
            # feat_train = np.hstack((bi_feat_train, xgb_feat_train))
            # feat_val = np.hstack((bi_feat_val, xgb_feat_val))

            # print(feat_train.shape)


            # feat_train = np.delete(feat_train, np.s_[80:], axis=1)
            # feat_val = np.delete(feat_val, np.s_[80:], axis=1)

            feat_train = np.delete(feat_train, np.s_[60:], axis=1)
            feat_val = np.delete(feat_val, np.s_[60:], axis=1)

            # clf = RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_leaf=1,
            #                              random_state=18, max_features='log2', n_jobs=-1)
            # clf = MultinomialNB(alpha=0.1)
            # clf = svm.SVC(C=0.1, gamma='auto', kernel='rbf', random_state=None, shrinking=True, tol=0.01)
            # clf = KNeighborsClassifier(n_neighbors=10, p=2, n_jobs=-1)
            # clf = AdaBoostClassifier(n_estimators=1500)
            # clf = svm.LinearSVC(penalty='l2', dual=False, C=0.4, multi_class='ovr', tol=1e-4)


            # model = xgb.train(params, xgb.DMatrix(trainx, trainy))  # ,num_boost_round=20)
            # y_predict = model.predict(xgb.DMatrix(valx))

            # print(y_predict.shape)


            # clf.fit(trainx, trainy)
            # y_predict = clf.predict(valx)

            # feat_perf = f.feature_subset_test(X, Y, train_index, val_index,
            #                                   classifier=clf, selector=selector, n_list=range(10000, 200000, 10000))
            #
            # print(feat_perf)
            # break

            # sclf = StackingCVClassifier(classifiers=clfs,
            #                             meta_classifier=clf_st2,
            #                             random_state=18, cv=3)
            #
            # sclf.fit(trainx, trainy)
            # y_predict = sclf.predict(valx)

            clfs_st2 = list()
            clfs_st2.append(LogisticRegression(C=3e-2, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr'))
            # clfs_st2.append(svm.LinearSVC(penalty='l2', dual=False, C=2e-3, multi_class='ovr', tol=1e-4))
            clfs_st2.append(BernoulliNB(alpha=1))
            clfs_st2.append(KNeighborsClassifier(n_neighbors=20, p=2, n_jobs=-1))
            # clfs_st2.append(RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=12,
            #                                        random_state=18, max_features=1, n_jobs=-1))

            # params_st2 = {'booster': 'gbtree',
            #               'learning_rate': 0.3,
            #               'subsample': 0.8,
            #               'colsample_bytree': 1,
            #               'eta': 0.01,
            #               'max_depth': 3,
            #               'min_child_weight': 3,
            #               'num_boost_round': 40,
            #               'objective': 'multi:softmax',
            #               'gamma': 3,
            #               'reg_lambda': 3e-5,
            #               # 'reg_alpha': 1e-5,
            #               'random_state': 19,
            #               'silent': 0,
            #               'num_class': 20
            #               }
            #
            # model_st2 = xgb.train(params_st2, xgb.DMatrix(feat_train, trainy))  # ,num_boost_round=20)
            # y_predict = model_st2.predict(xgb.DMatrix(feat_val))

            # clf_st2 = RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=12,
            #                                  random_state=18, max_features=1, n_jobs=-1)
            # clf_st2 = BernoulliNB(alpha=1000)
            # clf_st2 = KNeighborsClassifier(n_neighbors=20, p=2, n_jobs=-1)
            # clf_st2 = svm.LinearSVC(penalty='l2', dual=False, C=2e-3, multi_class='ovr', tol=1e-4)
            # clf_st2 = svm.SVC(C=30, gamma='auto', kernel='rbf', random_state=None, shrinking=True, tol=0.001)
            clf_st2 = LogisticRegression(C=5e-2, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr')

            clf_st2.fit(feat_train, np.ravel(trainy))
            y_predict = clf_st2.predict(feat_val)

            # cvst_st2 = f.Fold4CVStacking(clfs_st2, bi_feat_train, trainy)

            # bi_feat_train_st3 = cvst_st2.fit_train(n_fold=k, n_stage=2, reprocess=False)
            # bi_feat_val_st3 = cvst_st2.predict_val(bi_feat_val, n_fold=k, n_stage=2, reprocess=False)

            



            clf_st3 = LogisticRegression(C=0.05, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr')
            # clf_st3 = BernoulliNB(alpha=1)
            # clf_st3 = svm.LinearSVC(penalty='l2', dual=False, C=8e-2, multi_class='ovr', tol=1e-4)


            # clf_st3.fit(bi_feat_train_st3, np.ravel(trainy))
            # y_predict = clf_st3.predict(bi_feat_val_st3)


            # accuracy_SVM[k] = accuracy_score(valy_sub, y_predict_sub)


            accuracy_SVM[k] = accuracy_score(valy, y_predict)
            # accuracy_SVM[k] = accuracy_score(valy_meta, y_predict)

            print(accuracy_SVM[k])
            k += 1

        print('K-fold score', np.array(accuracy_SVM).mean())
        print('Runtime', time.time()-time_start, 's')
    else:

        test_tokenized_list = list()
        file_path = 'reddit_test.csv'
        n_lemm = 1
        test_token_path = 'Test_Tokens_' + str(n_lemm) + '_v4.json'

        if re_preprocess_test:
            with open(file_path, encoding='utf-8', mode='r') as csv_file:
                data_reader = csv.reader(csv_file, delimiter=',')

                feature_name = next(data_reader)
                i = 0
                for item in data_reader:
                    test_tokenized_list.append(f.text_preprocess(item[1], n_lemm, stem=None))
                    i = i + 1
                    if i % 1000 == 0: print(i)
            with open(test_token_path, 'w', encoding='utf-8') as file:
                data_save = [test_tokenized_list]
                json.dump(data_save, file, ensure_ascii=False)
        else:
            with open(test_token_path, 'r', encoding='utf-8') as file:
                test_tokenized_list = json.load(file)[0]

        cvc = CountVectorizer(stop_words=stopwords.words('english'), tokenizer=f.itself, preprocessor=None,
                              lowercase=False, max_df=1.0, min_df=2,
                              ngram_range=(1, 2), binary=False)  # 0.000014 is minimum min_df
        idftrans = TfidfTransformer()
        epf = f.expand_feature()
        selector = SelectKBest(chi2, k=30000)

        X = cvc.fit_transform(X)
        print(len(cvc.get_feature_names()))
        X = idftrans.fit_transform(X)
        X = epf.fit_transform(X, tokenized_list)
        X = selector.fit_transform(X, Y)

        X_test = cvc.transform(test_tokenized_list)
        X_test = idftrans.transform(X_test)
        X_test = epf.transform(X_test, test_tokenized_list)
        X_test = selector.transform(X_test)

        clfs = list()
        clfs.append(svm.LinearSVC(penalty='l2', dual=False, C=0.4, multi_class='ovr', tol=1e-4))
        clfs.append(MultinomialNB(alpha=0.1))
        clfs.append(LogisticRegression(C=5, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr'))
        clfs.append(RandomForestClassifier(n_estimators=1500, max_depth=120, min_samples_leaf=1,
                                           random_state=18, max_features='log2', n_jobs=-1))

        cvst = f.Fold2CVStacking(clfs, X, Y)

        bi_feat_train = cvst.fit_train(n_fold='test', n_stage=1, reprocess=True)
        bi_feat_val = cvst.predict_val(X_test, n_fold='test', n_stage=1, reprocess=True)

        feat_train = bi_feat_train
        feat_val = bi_feat_val

        # params = {'booster': 'gblinear',
        #           'learning_rate': 0.05,
        #           'eta': 0.01,
        #           'max_depth': 20,
        #           'min_child_weight': 1,
        #           'num_boost_round': 30,
        #           'objective': 'multi:softmax',
        #           'gamma': 0,
        #           'lambda': 3e-6,
        #           'random_state': 19,
        #           'silent': 0,
        #           'num_class': 20
        #           }
        # xgb_feat_train = cvst.fit_train_xgb(params, n_fold='test', n_stage=1, reprocess=True)
        # xgb_feat_val = cvst.predict_val_xgb(X_test, n_fold='test', n_stage=1, reprocess=True)

        # feat_train = np.hstack((bi_feat_train, xgb_feat_train))
        # feat_val = np.hstack((bi_feat_val, xgb_feat_val))

        # feat_train = np.delete(feat_train, np.s_[40:60], axis=1)
        # feat_val = np.delete(feat_val, np.s_[40:60], axis=1)


        # params_st2 = {'booster': 'gbtree',
        #               'learning_rate': 0.3,
        #               'subsample': 0.8,
        #               'colsample_bytree': 1,
        #               'eta': 0.01,
        #               'max_depth': 3,
        #               'min_child_weight': 3,
        #               'num_boost_round': 40,
        #               'objective': 'multi:softmax',
        #               'gamma': 3,
        #               'reg_lambda': 3e-5,
        #               # 'reg_alpha': 1e-5,
        #               'random_state': 19,
        #               'silent': 0,
        #               'num_class': 20
        #               }
        #
        # model_st2 = xgb.train(params_st2, xgb.DMatrix(feat_train, Y))  # ,num_boost_round=20)
        # Y_test = model_st2.predict(xgb.DMatrix(feat_val))

        clf_st2 = LogisticRegression(C=5e-2, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr')

        clf_st2.fit(feat_train, np.ravel(Y))
        Y_test = clf_st2.predict(feat_val)

        save_path = "test.csv"
        with open(save_path, 'w', newline='') as file:
            csv_write = csv.writer(file)
            csv_write.writerow(['Id', 'Category'])
            for ID, y_test_pred in zip(range(Y_test.size), Y_test):
                csv_head = [ID, inverse_label[y_test_pred]]
                csv_write.writerow(csv_head)

        print('Results written')













        




