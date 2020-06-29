# -*- coding: utf-8 -*-

# %%  import packages
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
#from mlxtend.classifier import StackingCVClassifier
import time
#import xgboost as xgb

import csv
import json

from Functions import metaclass_label
from Functions import class_label
inverse_label = {v: k for k, v in class_label.items()}


# %% pre-processing
model = 'BNB'
#  set model to : 'BNB' -> Bernoulli Naive Bayes; 
# 'LR' -> Logistic Regression
# 'RF' -> Random Forest
# 'Hard Voting' -> simple stacking model with hard voting 

re_preprocess = False    # pre-processed data has been stored in 'Tokens.json' file
testing = False    # set 'testing' to True for generating the prediction CSV file 
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

#%%  Training & Cross Validation
    if not testing:

        fold = 5
        kf = KFold(n_splits=fold, random_state=13839, shuffle=True)  # random state = 13839 for Group 23
        accuracy = np.zeros(fold)
        k = 0
        start_time = time.time()
        
        # cross-validation pipeline 
        for train_index, val_index in kf.split(X):

            trainx_list, valx_list = [X[i] for i in train_index], [X[i] for i in val_index]  # corresponding rows
            trainy, valy = Y[train_index], Y[val_index]
            trainy_meta, valy_meta = Y_meta[train_index], Y_meta[val_index]
            
            # feature extraction
            cvc1 = CountVectorizer(stop_words=stopwords.words('english'), tokenizer=f.itself, preprocessor=None,
                                  lowercase=False, max_df=1.0, min_df=0.00002,
                                  ngram_range=(1, 1), binary=True) 
            cvc2 = CountVectorizer(stop_words=stopwords.words('english'), tokenizer=f.itself, preprocessor=None,
                                  lowercase=False, max_df=1.0, min_df=0.00002,
                                  ngram_range=(1, 2), binary=True) 
           
            idftrans = TfidfTransformer()  # TF-IDF
            epf = f.expand_feature()  # add the length of the remark as a new feature
            selector = SelectKBest(chi2, k=15000) # feature selection

          
            # choose a classifier 
            if model == 'BNB':
                clf = f.bernoulli_NB()         # Bernoulli Naive Bayes
           
                trainx = cvc1.fit_transform(trainx_list)

                valx = cvc1.transform(valx_list)
            
            else:
                trainx = cvc2.fit_transform(trainx_list)

                valx = cvc2.transform(valx_list)
                
                trainx = idftrans.fit_transform(trainx)
                trainx = epf.fit_transform(trainx, trainx_list)
                trainx = selector.fit_transform(trainx, trainy)

            
                valx = idftrans.transform(valx)
                valx = epf.transform(valx, valx_list)
                valx = selector.transform(valx)
                
                if model == 'SVM':            # Support Vector Machine
                    clf = svm.LinearSVC(penalty='l2', dual=False, C=1, multi_class='ovr', tol=1e-4)
                elif model == 'LR':      # Logistic Regression
                    clf = LogisticRegression(C=5, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr')
                elif model == 'RF':       # Random Forest 
                    clf = RandomForestClassifier(n_estimators=1500, max_depth=120, min_samples_leaf=1,
                                          random_state=15, max_features='log2', n_jobs=-1)
                
                else:  # stacking wiht Hard Voting 
                    clf1 = svm.LinearSVC(penalty='l2', dual=False, C=0.4, multi_class='ovr', tol=1e-4) 
           
            
                    clf2 = MultinomialNB(alpha=0.1)           
                    clf3 = LogisticRegression(C=5, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr') 
                    clf4 = RandomForestClassifier(n_estimators=1500, max_depth=120, min_samples_leaf=1,
                                          random_state=15, max_features='log2', n_jobs=-1)
                   
                    clf = VotingClassifier(estimators=[('svm', clf1), ('MultiNB', clf2), ('rf',clf),('lr', clf3)], weights=[2, 2, 2, 1])
            
            
            
            clf.fit(trainx, trainy)
            y_predict = clf.predict(valx)
            accuracy[k] = accuracy_score(valy, y_predict)
    

            print(accuracy[k])
            k += 1

        print('K-fold score', np.array(accuracy).mean())
        print('Time Elapsed: %f seconds' % ((time.time() - start_time)/5))
    else:
        
        
# %%  testing 
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
        clfs.append(RandomForestClassifier(n_estimators=1200, max_depth=120, min_samples_leaf=1,
                                           random_state=15, max_features='log2', n_jobs=-1))

        cvst = f.Fold4CVStacking(clfs, X, Y)

        bi_feat_train = cvst.fit_train(n_fold='test', n_stage=1, reprocess=True)
        bi_feat_val = cvst.predict_val(X_test, n_fold='test', n_stage=1, reprocess=True)

        params = {'booster': 'gblinear',
                  'learning_rate': 0.05,
                  'eta': 0.01,
                  'max_depth': 20,
                  'min_child_weight': 1,
                  'num_boost_round': 30,
                  'objective': 'multi:softprob',
                  'gamma': 0,
                  'lambda': 3e-6,
                  'random_state': 19,
                  'silent': 0,
                  'num_class': 20
                  }
        xgb_feat_train = cvst.fit_train_xgb(params, n_fold='test', n_stage=1, reprocess=True)
        xgb_feat_val = cvst.predict_val_xgb(X_test, n_fold='test', n_stage=1, reprocess=True)

        feat_train = np.hstack((bi_feat_train, xgb_feat_train))
        feat_val = np.hstack((bi_feat_val, xgb_feat_val))

        feat_train = np.delete(feat_train, np.s_[40:60], axis=1)
        feat_val = np.delete(feat_val, np.s_[40:60], axis=1)

        params_st2 = {'booster': 'gbtree',
                      'learning_rate': 0.3,
                      'subsample': 0.8,
                      'colsample_bytree': 1,
                      'eta': 0.01,
                      'max_depth': 3,
                      'min_child_weight': 3,
                      'num_boost_round': 40,
                      'objective': 'multi:softmax',
                      'gamma': 3,
                      'reg_lambda': 3e-5,
                      # 'reg_alpha': 1e-5,
                      'random_state': 19,
                      'silent': 0,
                      'num_class': 20
                      }

        model_st2 = xgb.train(params_st2, xgb.DMatrix(feat_train, Y))  # ,num_boost_round=20)
        Y_test = model_st2.predict(xgb.DMatrix(feat_val))


        save_path = "test.csv"
        with open(save_path, 'w', newline='') as file:
            csv_write = csv.writer(file)
            csv_write.writerow(['Id', 'Category'])
            for ID, y_test_pred in zip(range(Y_test.size), Y_test):
                csv_head = [ID, inverse_label[y_test_pred]]
                csv_write.writerow(csv_head)

        print('Results written')













        




