#!/usr/bin/env python
# coding: utf-8

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import datasets
from pandas import DataFrame
import nltk
import numpy as np
import nltk.stem
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
from sklearn.linear_model import LogisticRegression
import os
from sklearn.svm import SVC
from numpy import random


def load_text(pos_path, neg_path):
    #pos_path = 'rt-polarity.pos'
    #neg_path = 'rt-polarity.neg'
    f_pos = open(pos_path, 'r', encoding='ISO-8859-1')
    f_neg = open(neg_path, 'r', encoding='ISO-8859-1')
    
    all_sentences = []
    for sentence in sent_tokenize(f_pos.read()):
        all_sentences.append({'context': sentence, 'sentiment': 'positive'})
    for sentence in sent_tokenize(f_neg.read()):
        all_sentences.append({'context': sentence, 'sentiment': 'negative'})
     
    f_pos.close()
    f_neg.close()
    
    return all_sentences




def create_df(sentences_with_class):
    sent = DataFrame(sentences_with_class)
    return sent




# generate all possible combinations of parameters (including default)
def generate_parameter():
    list_of_parameters = [['lemm', 'stem', None], # use lemmatize or stem
                         [None, 'english'], # use stopword or not
                         [0, 0.1, 0.01] # min_df
                         ]
    para_groups = list(itertools.product(*list_of_parameters))
    return para_groups



# analyzer for use in CountVectorizer
stemmer = SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
lemmatizer = nltk.stem.WordNetLemmatizer()
class LemmCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])




def random_model(test_X):
    random_results = []
    two_outcomes = ['positive', 'negative']
    for line in test_X:
        random_results.append(random.choice(two_outcomes))
    return random_results


def run_Classifier(para_groups, clfs, train):
    
    
    # set up all containers for test results
    all_scores = []
    all_paras = []
    all_clf_pipes = []
    #all_conf_mat = []
    
    with open('accuracy_summarys_final.txt', 'a+') as f:
    
        current_iter = 1
        
        for clf in clfs:
            f.write('\n---------------------------------------------\n')
            f.write('Current experimenting type of classifier: ' + str(clf) + ' \n\n\n')
            
            for paras in para_groups:
                
                print('current iteration: ' + str(current_iter))
                current_iter = current_iter + 1
                
                if (paras[0] == 'lemm'): 
                    vectorizer = LemmCountVectorizer(min_df=paras[2], analyzer="word", stop_words=paras[1])
                if (paras[0] == 'stem'):
                    vectorizer = StemmedCountVectorizer(min_df=paras[2], analyzer="word", stop_words=paras[1])
                else:
                    vectorizer = CountVectorizer(min_df=paras[2], stop_words=paras[1])
        
                pipe = Pipeline([
                    ('count_vectorizer', vectorizer),
                    ('classifier', clf)
                    ])
                
                kf_clf_scores = []
                kf_random_scores = []
        
                # use cross-validation
                X = train.context
                kf = KFold(n_splits= 8, shuffle=True)
                for train_index, test_index in kf.split(X):
                    train_X = df.iloc[train_index]['context'].values
                    train_Y = df.iloc[train_index]['sentiment'].values
                    test_X = df.iloc[test_index]['context'].values
                    test_Y = df.iloc[test_index]['sentiment'].values
                    
                    pipe.fit(train_X, train_Y)
                    random_Y = random_model(test_X)
                    
                    current_clf_score = accuracy_score(test_Y, pipe.predict(test_X))
                    random_score = accuracy_score(test_Y, random_Y)
                
                    kf_clf_scores.append(current_clf_score)
                    kf_random_scores.append(random_score)
                    
                    
                all_scores.append(sum(kf_clf_scores)/len(kf_clf_scores))
                all_paras.append(paras)
                all_clf_pipes.append(pipe)
                    
                f.write('Parameters: ' + str(paras) + '\n')
                f.write('Score: ' + str(sum(kf_clf_scores)/len(kf_clf_scores)) + '\n')
                f.write('Random Model Score: ' + str(sum(kf_random_scores)/len(kf_random_scores)) + '\n\n')

        

    
    return all_scores, all_paras, all_clf_pipes
    



def find_best_model(all_scores, all_paras, pipes, clfs, para_groups):
    
    best_clf_pipes = []
    best_paras = []
    
    with open('best_models_final.txt', 'a+') as best:
    
        i = 0
        for clf in clfs:
            subarr_acc = all_scores[i : i+len(para_groups)-1]
            best_acc = max(subarr_acc)
            best_idx = subarr_acc.index(max(subarr_acc))
            
            i = i + len(para_groups)
            
            best.write('The best ' + str(clf) + ' used the following parameters: ' + str(all_paras[best_idx]) + ' \n')
            best.write('Train Accuracy: ' + str(best_acc) + '\n')
            best.write('--------------------------------------\n')
            
            best_paras.append(all_paras[best_idx])
            best_clf_pipes.append(pipes[best_idx])
    
    return best_clf_pipes, best_paras
    



# this function has been discarded
def test_best_model(clf_pipes, paras, test_df):
    
    clfs = [BernoulliNB(), LogisticRegression(solver='lbfgs', max_iter=400), SVC(kernel = 'linear'), MultinomialNB()]
    
    with open('best_models_final.txt', 'a+') as best:
        
        best.write('\n\n\nBelow are test accuracies and confusion matrices of the above best models: \n\n')
    
        for i in range(len(clf_pipes)):
            Y_predict = clf_pipes[i].predict(test_df.context)
            model_acc = accuracy_score(test_df.sentiment, Y_predict)
            con_m = confusion_matrix(test_df.sentiment, Y_predict)
            
            best.write('Best ' + str(clfs[i]) + ' \n')
            best.write('Test Accuracy: ' + str(best_acc) + '\n')
            best.write('Confusion matrix: ' + str(con_m) )
            best.write('--------------------------------------\n')
            


# run_test_set is similar to run_Classifier but without implementation of cross-validation
def run_test_set(clfs, para_groups, training_set_df, test_set_df):
    
    with open('best_models_test_set_final.txt', 'a+') as best:
        
        for i in range(len(clfs)):
            print('current iteration: ' + str(i))
            
            paras = para_groups[i]
            
            if (paras[0] == 'lemm'): 
                vectorizer = LemmCountVectorizer(min_df=paras[2], analyzer="word", stop_words=paras[1])
            if (paras[0] == 'stem'):
                vectorizer = StemmedCountVectorizer(min_df=paras[2], analyzer="word", stop_words=paras[1])
            else:
                vectorizer = CountVectorizer(min_df=paras[2], stop_words=paras[1])
            
            
            pipe = Pipeline([
                    ('count_vectorizer', vectorizer),
                    ('classifier', clfs[i])
                    ])
            
            pipe.fit(training_set_df.context,training_set_df.sentiment)
            test_y_hat = pipe.predict(test_set_df.context)
            test_acc = accuracy_score(test_set_df.sentiment, test_y_hat)
            
            conf_mtx = confusion_matrix(test_set_df.sentiment, test_y_hat, labels = ['positive', 'negative'])
            
            best.write(str(clfs[i]) + ' with parameters: ' + str(para_groups[i]) + 
                       ' has test accuracy: ' + str(test_acc) + '\n')
            best.write('Its confusion matrix is\n' + str(conf_mtx) + '\n\n')
            
            

df = create_df(load_text(r'rt-polarity.pos', r'rt-polarity.neg'))

para_groups = generate_parameter()

train_df, test_df = train_test_split(df, test_size=0.2)

clfs = [BernoulliNB(), LogisticRegression(solver='lbfgs', max_iter=400), SVC(kernel = 'linear'), MultinomialNB()]

all_scores, all_paras, all_clf_pipes = run_Classifier(para_groups, clfs, train_df)

find_best_model(all_scores, all_paras, all_clf_pipes, clfs, para_groups)

print('Running clfs on test set...')

# manually examine the best clfs and parameters in 'best_models.txt' and typed the data below
best_para_groups = [('stem', None, 0) , ('stem', None, 0), ('stem', None, 0), ('lemm', None, 0) ]

run_test_set(clfs, best_para_groups, train_df, test_df)

