# -*- coding: utf-8 -*-
"""
@author: Luiz Resende Silva - ID 260852243
"""
##################################################################
'''                 GENERAL LIBRARIES                       '''
##################################################################
import pandas as pd
import numpy as np
import timeit
from joblib import Parallel, delayed
##################################################################
'''                     CLASSIFIERS                        '''
##################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
##################################################################
'''      STACKING CROSS-VALIDATION META-CLASSIFIE            '''
##################################################################
from mlxtend.classifier import StackingCVClassifier
##################################################################
'''           PY FILE CONTAINING FUNCTIONS BUILT             '''
##################################################################
import Project_02_Functions as pf
from Project_02_UserDefinedClasses import CustomStackVoting
from Project_02_UserDefinedClasses import MultiClassBernoulliNB
##################################################################################################################################
'''                                                  REDDIT TESTING DATA                                                   '''
##################################################################################################################################
FileTrain = "reddit_train.csv" #Should contain the correct path to the training dataset file
dataTrainRead = pf.Read_File_DF(FileTrain, separation=',', head=0, replace=None, drop=False) #Reading file

FileTest = "reddit_test.csv" #Should contain the correct path to the test dataset file
dataTestRead = pf.Read_File_DF(FileTest, separation=',', head=0, replace=None, drop=False) #Reading file

dataTrain = dataTrainRead.copy(deep=True)
dataTest = dataTestRead.copy(deep=True)

""" Preprocessing training and testing data """
preProcessTrain = []
for doc in dataTrain['comments']:
    preProcessTrain.append(pf.Document_Preprocess(Doc_Comment=doc, RemoveStop=True, DoLemma=True, DoSplit=False))
dataTrain['comments'] = preProcessTrain

preProcessTest = []
for doc in dataTest['comments']:
    preProcessTest.append(pf.Document_Preprocess(Doc_Comment=doc, RemoveStop=True, DoLemma=True, DoSplit=False))
dataTest['comments'] = preProcessTest

""" Counting and storing classes' names for future use """
clas = dataTrain["subreddits"].unique().tolist()

""" Defining features and outputs in the training dataset """
out_train = dataTrain["subreddits"]
features = dataTrain.iloc[:,[0,1]]

""" Possible different parameters to be used in the ExtractVectorizer """
#Stop = pf.StopW_NLTK(DicLan='english') + pf.StopW_Punct() OR 'english'
#Accents = 'ascii'
#Token = pf.LemmatizerTokens()
""" FEATURE EXTRACTION - Importing method to vectorize the features """
print("="*100)
vec_training, vec_testing = pf.ExtractVectorizer(DataTrain=dataTrain["comments"], DataTest=dataTest["comments"], CountVecXTFIDF=False,
                                                 Accents='unicode', Token=None, Stop=None, nGram=(1,1), Binar=False, Regular='l1',
                                                 Normal=True, SubLinear=True, MinDf=1, AdditFeat=False, FeatTrainToAdd=None,
                                                 FeatTestToAdd=None, show=False)
print("="*100)
print("Feature extraction vectorization: Done")
print("Dimensionality of vectors:",vec_training.shape, vec_testing.shape)

#######################################################################
"         INSTANTIATING LIST OF OBJECT TYPES CLASSIFIERS           "
#######################################################################
LogReg = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, class_weight=None,
                            random_state=None, solver='saga', max_iter=1500, multi_class='auto', verbose=2, warm_start=False, n_jobs=-1, l1_ratio=0.75)

kSVC = SVC(kernel='poly', C=30, degree=3, gamma='auto', probability=True, tol=0.001, max_iter=500, decision_function_shape='ovr', cache_size=500, verbose=0,
           random_state=None)

lSVC = LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                 multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,verbose=1)

SGDC = SGDClassifier(alpha=0.000001, average=False, class_weight=None, early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                     l1_ratio=0.75, learning_rate='optimal', loss='log', max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='elasticnet',
                     power_t=0.5, random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1, verbose=2, warm_start=False)

BNB = BernoulliNB(alpha=0.01, class_prior=None, fit_prior=True)

MNB = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

RandForest = RandomForestClassifier(n_estimators=2000, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                                    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=-1, oob_score=False,
                                    random_state=None, verbose=1, warm_start=False)

ExTrees = ExtraTreesClassifier(n_estimators=1200, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=1,
                               warm_start=False, class_weight=None)

GradBoosting = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0, criterion='friedman_mse',
                                          min_samples_split=2, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_depth=3,
                                          min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None,
                                          max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, presort='auto',
                                          validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

AdaBoost = AdaBoostClassifier(base_estimator=kSVC, n_estimators=2, learning_rate=1.0, algorithm='SAMME', random_state=None)

MCBnb = MultiClassBernoulliNB() #Model constructed from scratch

#######################################################################
"          DEFINING LIST OF CLASSIFIERS AVAILABLE TO BE USED          "
#######################################################################
ListAllClassifiers = [("LogisticRegression", LogReg) #0
                   ,("kernelSVC", kSVC) #1
                   ,("LinearSVC", lSVC) #2
                   ,("SGD_Classifier", SGDC) #3
                   ,("Bernoulli_Naive_Bayes", BNB) #4
                   ,("Multinomial_Naive_Bayes", MNB) #5
                   ,("Random_Forest", RandForest) #6
                   ,("Extra_Trees", ExTrees) #7
                   ,("Gradient_Boosting", GradBoosting) #8
                   ,("AdaBoostClassifier", AdaBoost) #9
                   ,("Bernoulli_Scratch", MCBnb) #10
                   ]

#######################################################################
"             DEFINING LIST OF CUSTOM VOTING CLASSIFIER             "
#######################################################################
ListCustomVoting = [("LogisticRegression_V", LogReg) #0
                   ,("SGD_Classifier_V", SGDC) #1
                   ,("Bernoulli_Naive_Bayes_V", BNB) #2
                   ,("Random_Forest_V", RandForest) #3
                   ,("Multinomial_Naive_Bayes_V", MNB) #4
                   ,("ExtraTrees_V", ExTrees) #5
                   ,("Gradient_Boosting_V", GradBoosting) #6
                   ,("AdaBoostClassifier_V", AdaBoost) #7
                    ]
CustomVoting = CustomStackVoting(ListCustomVoting, voting='majority', n_jobs=-1, verbose=1)
EnsembleCustom = [("Ensemble Custom Voting Classifier", CustomVoting)]

#######################################################################
"         DEFINING LIST OF ENSEMBLE STACKING META-CLASSIFIER         "
#######################################################################
ListModelMetaClassifier = [MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True) #0
                    ,BernoulliNB(alpha=0.01, class_prior=None, fit_prior=True) #1
                    ,RandomForestClassifier(n_estimators=3000, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                          max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=2,
                          min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False) #2
                    ,ExtraTreesClassifier(n_estimators=1500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                         min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                         min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None) #
                    ,GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=300, subsample=1.0,
                                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=2, 
                                                min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                                min_impurity_split=None, init=None, random_state=None, max_features=None,
                                                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
                                                validation_fraction=0.1, n_iter_no_change=None, tol=0.0001) #3
                    ,AdaBoostClassifier(base_estimator=kSVC, n_estimators=10, learning_rate=1.0, algorithm='SAMME', random_state=None) #4
                   ]


Meta_Estimator = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1, class_weight=None,
                                    random_state=None, solver='saga', max_iter=1500, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1,
                                    l1_ratio=0.75)


MetaClass = StackingCVClassifier(classifiers=ListModelMetaClassifier, meta_classifier=Meta_Estimator, use_probas=True, drop_last_proba=False, cv=3,
                                 shuffle=False, random_state=None, stratify=True, verbose=2, use_features_in_secondary=False, 
                                 store_train_meta_features=False, use_clones=True, n_jobs=-1)

#####################################################################################################################################################
"""             FITTING THE MODEL AND MAKE PREDICTIONS - SET THE FLAGS BELOW TO true OR false DEPENDING ON THE SELECTED APPROACH              """
#####################################################################################################################################################
SINGLE = False
ALL = False
META = True

print("START FITTING....")
print("="*100)
tStart = timeit.default_timer()

if(SINGLE):
    if(ALL):
        HeldOutDataPredictions = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(pf.Classification_Model)(data_training=vec_training, target_training=out_train,
                           data_testing=vec_testing, Classifier=Model[1], target_testing=None, ModelName=Model[0], accur=False, grph=False, setClass=clas, show=False) for Model in ListAllClassifiers)

    else:
        HeldOutDataPredictions = pf.Classification_Model(data_training=vec_training, target_training=out_train, data_testing=vec_testing,
                                    Classifier=ListAllClassifiers[10][1], target_testing=None, ModelName=ListAllClassifiers[10][0],
                                    accur=False, grph=False, setClass=clas, show=False)
elif(META==False):
    HeldOutDataPredictions = pf.Classification_Model(data_training=vec_training, target_training=out_train, data_testing=vec_testing,
                                    Classifier=EnsembleCustom[0][1], target_testing=None, ModelName=EnsembleCustom[0][0],
                                    accur=False, grph=False, setClass=clas, show=False)
else:
    MetaClass.fit(vec_training, out_train)
    HeldOutDataPredictions = MetaClass.predict(vec_testing)

runingTime = timeit.default_timer() - tStart #Stopping clock and getting time spent
print("Fitting and predictions done in %0.4fs."%runingTime)
print("="*100)

""" PRINTING THE PREDICTIONS MADE AND SAVING CSV FILE """
Preds = pd.DataFrame({"Category":HeldOutDataPredictions})
Results = pd.concat([dataTest["id"], Preds], axis=1, sort=False) 
print(Results)
pf.Write_File_DF(Data_Set=Results, File_Name="Predictions_Group_4", separation=",", head=True, ind=False)

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
##################################################################################################################################
