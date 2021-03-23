# -*- coding: utf-8 -*-
"""
@author: Luiz Resende Silva
"""
##################################################################
'''                 GENERAL LIBRARIES                       '''
##################################################################
import pandas as pd
import numpy as np
##################################################################
'''                 FEATURE EXTRACTION                       '''
##################################################################
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
##################################################################
'''                 FEATURE SELECTION                       '''
##################################################################
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
##################################################################
'''                     CLASSIFIERS                        '''
##################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
##################################################################
'''           PIPELINE, GRID SEARCH AND METRICS             '''
##################################################################
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
##################################################################
'''      STACKING CROSS-VALIDATION META-CLASSIFIE            '''
##################################################################
from mlxtend.classifier import StackingCVClassifier
##################################################################
'''           PY FILE CONTAINING FUNCTIONS BUILT             '''
##################################################################
import Project_02_Functions as pf

##################################################################################################################################
'''                                                 STARTING                                                    '''
##################################################################################################################################

" LOADING THE DATASET "
FileTrain = "reddit_train.csv"
DataSet = pf.Read_File_DF(FileTrain, separation=',', head=0, replace=None, drop=False)
print("Data loaded\n")

#process = []
#for doc in DataSet['comments']:
#    process.append(pf.Document_Preprocess(Doc_Comment=doc, LowerCase=True, RemoveHTML=False, StripAccent=True, Accented='ascii',
#                                          StripCharSpec=True, RemoveStop=True, StopWords=(pf.StopW_NLTK('english')), DoLemma=True,
#                                          DoSplit=False))
#DataSet['comments'] = process

" DIVIDING THE DATA SET INTO TRAINING AND TESTING FOR FURTHER VALIDATION "
data_train, data_valid, target_train, target_valid = train_test_split(DataSet.iloc[:,[0,1]], DataSet["subreddits"], test_size=0.30,
                                                                    random_state=106750, shuffle=True, stratify=DataSet["subreddits"])
#print(data_train, data_valid, target_train, target_valid)

print("Data splitted in two subsets: training and validation to decrease size of datasets")

" DEFINING THE PARAMETERS TO BE USED IN THE GRID SEARCH FOR THE FEATURE EXTRACTION/SELECTION "
ParametersGrid = [{#'vectorizer': [TfidfVectorizer(decode_error='strict', strip_accents='unicode', lowercase=True, preprocessor=None,
#                                                  tokenizer=None, analyzer='word', stop_words='english',
#                                                  ngram_range=(1, 1), max_df=1.0, max_features=None, vocabulary=None,
#                                                  binary=False, norm='l1', use_idf=True, smooth_idf=True, sublinear_tf=True)],
#         'vectorizer__stop_words': (None, 'english'),
#         'vectorizer__strip_accents': (None, 'unicode'),
#         'vectorizer__tokenizer': (None, pf.LemmatizerTokens()),
#         'vectorizer__binary': (False, True),
#         'vectorizer__norm': ('l1', 'l2', None),
#         'vectorizer__ngram_range': ((1, 2), (2,2), (1,3)),  # unigrams or bigrams
#         'vectorizer__min_df': (2, 3),
#         'vectorizer__sublinear_tf': (True, False),
#         'selector': [SelectKBest(chi2), SelectKBest(mutual_info_classif)],
#         'selector__k':  (5000, 30000, 'all'),
#         'selector': [TruncatedSVD(random_state=106750)],
#         'selector__n_components': (100, 1000, 10000),
#         'selector__n_iter':(5, 10, 25),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'selector': [NMF(random_state=106750, shuffle=True)],
#         'selector__n_components': (500, 5000, 50000, None),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'selector__max_iter':(200, 500),         
#         'classifier__alpha': (0.01, 0.5, 1.0),    #Bernoulli/Multinomial Naive-Bayes
#         'classifier__fit_prior': (True, False)
#         'classifier__binarize': (True, False),
#         'classifier__penalty': ('elasticnet', 'l1', 'l2'),    #Logistic Regression
#         'classifier__l1_ratio': (0.75, 1.0),
#         'classifier__max_iter': (500, 1500),
#         'classifier__kernel': ('rbf', 'poly', 'sigmoid'),    #SVC
#         'classifier__C': (0.1, 1, 10),
#         'classifier__degree': (3, 4),
#         'classifier__gamma': ('auto', 'scale'),
#         'classifier__tol': (0.001, 0.0001, 0.00001),
#         'classifier__max_iter': (2000, 5000),
#         'classifier__decision_function_shape': ('ovo', 'ovr'),
#         'classifier__random_state': (None, 106750),
#         'classifier__penalty': ('l1', 'l2'),    #Linear SVC
#         'classifier__C': (0.1, 1, 10),
#         'classifier__max_iter': (1000, 2000),
#         'classifier__penalty': ('l1', 'l2', 'elasticnet'),    #SGD Classifier
#         'classifier__alpha': (0.0001, 0.00001, 0.000001),
#         'classifier__max_iter': (1000, 2000),
#         'classifier__learning_rate': ('optimal', 'adaptive'),
#         'classifier__n_neighbors': (10, 15, 25),  #kNN
#         'classifier__p': (1, 2),  #Random Forest
#         'classifier__n_estimators': (500, 1000, 2000),
#         'classifier__n_estimators': ('gini', 'entropy')
        }]

" DEFINING LIST OF CLASSIFIERS AVAILABLE TO BE USED - UNCOMMENT LINES IN THE LIST OF ESTIMATIORS TO ENABLE MODELS FOR THE VOTING CLASSIFIER "
ListClassifiers = [#("LogisticRegression", LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=25, fit_intercept=True,
#                                                             intercept_scaling=1, class_weight=None, random_state=None, solver='saga',
#                                                             max_iter=1500, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1,
#                                                             l1_ratio=0.75))
#                   ,("kernelSVC", SVC(C=25, kernel='rbf', degree=2, gamma='scale', probability=True,  tol=0.001, cache_size=1000,
#                                     class_weight=None, verbose=False, max_iter=2500, decision_function_shape='ovr',random_state=None))
#                   ,("LinearSVC", LinearSVC(C=10, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
#                                           loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l1', random_state=None,
#                                           tol=0.0001,verbose=0))
#                   ,("SGD_Classifier", SGDClassifier(penalty='elasticnet', n_jobs=-1))
#                   ,("Bernoulli_Naive_Bayes", BernoulliNB(alpha=1.0))
                   ("Multinomial_Naive_Bayes", MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
#                   ,("kNearest_Neighbors", KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
#                                                                n_jobs=-1, n_neighbors=25, p=1, weights='uniform'))
#                   ,("Random_Forest", RandomForestClassifier(n_estimators=1000, bootstrap=True, class_weight=None, criterion='gini',
#                                                             max_depth=None, max_features='auto', max_leaf_nodes=None,
#                                                             min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=2,
#                                                             min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=-1, oob_score=False,
#                                                             random_state=None, verbose=0, warm_start=False))
                   ]

ListEstimatorsVoting = [#("LogisticRegression", LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10, fit_intercept=True,
#                                                             intercept_scaling=1, class_weight=None, random_state=None, solver='saga',
#                                                             max_iter=1500, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1,
#                                                             l1_ratio=0.75)) #0
#                   ,("LinearSVC", LinearSVC(C=10, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
#                                           loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l1', random_state=None,
#                                           tol=0.0001,verbose=0)) #1
#                   ,("SGD_Classifier", SGDClassifier(alpha=0.0001, max_iter=1000, penalty='elasticnet', l1_ratio=0.75, n_jobs=-1)) #2
                   ("Bernoulli_Naive_Bayes", BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)) #3
                   ,("Multinomial_Naive_Bayes", MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)) #4
#                   ,("kNearest_Neighbors", KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
#                                                                n_jobs=-1, n_neighbors=25, p=1, weights='uniform')) #5
#                   ,("Random_Forest", RandomForestClassifier(n_estimators=2000, bootstrap=True, class_weight=None, criterion='gini',
#                                                             max_depth=None, max_features='auto', max_leaf_nodes=None,
#                                                             min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=2,
#                                                             min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=-1, oob_score=False,
#                                                             random_state=None, verbose=0, warm_start=False)) #6
                    ] 

EnsembleVoting = [("Ensemble Voting Classifier", VotingClassifier(estimators=ListEstimatorsVoting, voting='hard', n_jobs=-1))]

" DEFINING LIST TO STORE RESULTS FROM EACH CLASSIFIER "
ClassifiersNames = []
ResultsAccuracy = []
ResultsTime = []

" ITERATING THROUGH THE LIST OF CLASSIFIERS AND TESTING THEM "
for name, model in ListClassifiers:
    print('=' * 100)
    print(name)
    Temp1, Temp2, Temp3, clf = pf.ClassificationModelGrid(Examples=data_train["comments"], Labels=target_train, Classifier=model, Name=name,
                                                     GridParameters=ParametersGrid, Folds=2, Normal=False, UseSelec=False, Ensemble=False, GetFitted=True)
    ClassifiersNames.append(Temp1)
    ResultsAccuracy.append(Temp2)
    ResultsTime.append(Temp3)
    print('=' * 100)
preds = clf.predict(data_valid["comments"])
print(("Model accuracy on held-out data: %0.4f")%(metrics.accuracy_score(target_valid, preds, normalize=True)))

#for name, model in EnsembleVoting:
#    print(name)
#    Temp1, Temp2, Temp3, clf = pf.ClassificationModelGrid(Examples=data_train["comments"], Labels=target_train, Classifier=model, Name=name,
#                                                     GridParameters=ParametersGrid, Folds=2, Normal=False, UseSelec=False, Ensemble=True, GetFitted=True)
#    ClassifiersNames.append(Temp1)
#    ResultsAccuracy.append(Temp2)
#    ResultsTime.append(Temp3)
#    print('=' * 100)
#preds = clf.predict(data_valid["comments"])
#print(("Model accuracy on held-out data: %0.4f")%(metrics.accuracy_score(target_valid, preds, normalize=True)))

" PRINTING GRAPHICAL COMPARISON OF ACCURACY AND TIME BETWEEN MODELS "
Results = pd.DataFrame({"Classifiers":ClassifiersNames, "Accuracy":(np.array(ResultsAccuracy)*100), "Run Time":ResultsTime})
pf.Learn_Perform(DataF=Results, LabelX='Classifiers', LabelY1='Accuracy', LabelY2='Run Time', TitleName="Resulting Scores", save=False)

##################################################################################################################################
'''                                        GRID SEARCH LISTS OF PARAMETERS                                                     '''
##################################################################################################################################
#ParametersGrid = [{'vectorizer': [CountVectorizer(), TfidfVectorizer(norm='l1')],
#         'vectorizer__stop_words': (None, (pf.StopW_NLTK('english')+pf.StopW_Punct()), 'english'),
#         'vectorizer__strip_accents': (None, 'unicode', 'ascii'),
#         'vectorizer__binary': (False, True),
#         'vectorizer__norm': ('l1', 'l2'),
#         'vectorizer__ngram_range': ((1, 1), (2, 2), (3,3)),  # unigrams or bigrams
#         'vectorizer__min_df': (1, 2, 3),
#         'vectorizer__sublinear_tf': (True, False),
#         'selector': [SelectKBest(chi2), SelectKBest(mutual_info_classif)],
#         'selector__k':  (5000, 30000, 'all'),
#         'selector': [TruncatedSVD(random_state=106750)],
#         'selector__n_components': (100, 1000, 10000),
#         'selector__n_iter':(5, 10, 25),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'selector': [NMF(random_state=106750, shuffle=True)],
#         'selector__n_components': (500, 5000, 50000, None),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'selector__max_iter':(200, 500),
#         'classifier__alpha': (0.01, 0.5, 1.0),    #Bernoulli/Multinomial Naive-Bayes
#         'classifier__fit_prior': (True, False)
#         'classifier__binarize': (True, False),
#         'classifier__penalty': ('elasticnet', 'l1', 'l2'),    #Logistic Regression
#         'classifier__l1_ratio': (0.0, 0.5, 0.75, 1.0),
#         'classifier__max_iter': (500, 1500),
#         'classifier__penalty': ('l1', 'l2'),    #Linear SVC
#         'classifier__C': (0.1, 1, 10),
#         'classifier__max_iter': (1000, 2000),
#         'classifier__penalty': ('l1', 'l2', 'elasticnet'),    #SGD Classifier
#         'classifier__alpha': (0.0001, 0.00001, 0.000001),
#         'classifier__max_iter': (1000, 2000),
#         'classifier__learning_rate': ('optimal', 'adaptive'),
#         'classifier__n_neighbors': (10, 15, 25),  #kNN
#         'classifier__p': (1, 2),  #Random Forest
#         'classifier__n_estimators': (500, 1000, 2000),
#         'classifier__n_estimators': ('gini', 'entropy')
#        }]
#ParametersGridVoter = [{'vectorizer': [TfidfVectorizer(), CountVectorizer()],
#         'vectorizer__stop_words': (None, (pf.StopW_NLTK('english')+pf.StopW_Punct()), 'english'),
#         'vectorizer__strip_accents': (None, 'unicode', 'ascii'),
#         'vectorizer__tokenizer': (None, pf.LemmatizerTokens()),
#         'vectorizer__binary': (False, True),
#         'vectorizer__ngram_range': ((1, 1), (2, 2), (3,3)),  # unigrams or bigram
#         'vectorizer__norm': ('l1', 'l2', None),
#         'vectorizer__sublinear_tf': (True, False),
#         'vectorizer__min_df': (1, 2, 3),
#         'selector': [SelectKBest(chi2), SelectKBest(mutual_info_classif)],
#         'selector__k':  (5000, 30000, 'all'),
#         'selector': [NMF(random_state=106750, shuffle=True)],
#         'selector__n_components': (500, 5000, 50000, None),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'selector__max_iter':(200, 500),
#         'selector__l1_ratio': (0.0, 0.5, 1.0),
#         'selector__alpha': (0.1, 0.5, 1.0),
#         'selector': [TruncatedSVD(random_state=106750)],
#         'selector__n_components': (100, 1000, 10000),
#         'selector__n_iter':(5, 10, 25),
#         'selector__tol':(0.001, 0.00001, 0.0),
#         'Bernoulli_Naive_Bayes__alpha': (0.01, 0.5, 1.0),
#         'Bernoulli_Naive_Bayes__fit_prior': (True, False)
#         'Bernoulli_Naive_Bayes__binarize': (True, False),
#         'Multinomial_Naive_Bayes__alpha': (0.01, 0.5, 1.0),
#         'Multinomial_Naive_Bayes__fit_prior': (True, False),
#         'LogisticRegression__penalty': ('elasticnet', 'l1', 'l2'),
#         'LogisticRegression__l1_ratio': (0.0, 0.5, 0.75, 1.0),
#         'LogisticRegression__max_iter': (500, 1500),
#         'LinearSVC__penalty': ('l1', 'l2'),
#         'LinearSVC__C': (0.1, 1, 10),
#         'LinearSVC__max_iter': (1000, 2000),
#         'SGD_Classifier__penalty': ('l1', 'l2', 'elasticnet'),
#         'SGD_Classifier__alpha': (0.0001, 0.00001, 0.000001),
#         'SGD_Classifier__max_iter': (1000, 2000),
#         'SGD_Classifier__learning_rate': ('optimal', 'adaptive'),
#         'kNearest_Neighbors__n_neighbors': (10, 15, 25),
#         'kNearest_Neighbors__p': (1, 2),
#         'Random_Forest__n_estimators': (500, 1000, 2000),
#         'Random_Forest__n_estimators': ('gini', 'entropy')
#            }]

##################################################################################################################################
'''                                DEFAULT PARAMETERS FOR THE DIFFERENT CLASSIFIERS AVAILABLE                                '''
##################################################################################################################################
"""
#Default parameters for LogisticRegression -> (penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
#                                         class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn',
#                                         verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
#
#Default parameters for LinearSVC -> (penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
#                                     intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
#
#Default parameters for SVC -> (C=1.0, kernel='rbf', degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False
#                               tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
#                               random_state=None)
#
#Default parameters for SGDClassifier -> (loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
#                                         tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
#                                         learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1,
#                                         n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
#
#Default parameters for Bernoulli Naive-Bayes -> (alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
#
#Default parameters for Multinomial Naive-Bayes -> (alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
#
#Default parameters for k-Nearest Neighbors -> (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
#                                               metric_params=None, n_jobs=None, **kwargs)
#
#Default parameters for RandomForestClassifier -> (n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2,
#                                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
#                                                  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
#                                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
#                                                  warm_start=False, class_weight=None)
#
#Default parameters for ExtraTreesClassifier -> (n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2,
#                                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None,
#                                                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False,
#                                                n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
#
#Default parameters for GradientBoostingClassifier -> (loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
#                                                      criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
#                                                      min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
#                                                      min_impurity_split=None, init=None, random_state=None, max_features=None,
#                                                      verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
#                                                      validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
#
#Default parameters for AdaBoostClassifier -> (base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
#
#Default parameters for VotingClassifier -> (estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True)
#
#Default parameters for StackingCVClassifier -> (classifiers, meta_classifier, use_probas=False, drop_last_proba=False, cv=2, shuffle=True,
#                                                random_state=None, stratify=True, verbose=0, use_features_in_secondary=False,
#                                                store_train_meta_features=False, use_clones=True, n_jobs=None, pre_dispatch='2n_jobs')
"""
##################################################################################################################################
'''                                                              END                                                           '''
##################################################################################################################################S
