# -*- coding: utf-8 -*-
"""
@author CustomStackVoting: Luiz Resende Silva
@author MultiClassBernoulliNB: Matheus Faria
"""
import pandas as pd
import numpy as np
import time
import timeit

import Project_02_Functions as pf

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from sklearn.utils import Bunch

##############################################################################################################################
"""                                         CLASS MULTICLASS BERNOULLI NAIVE BAYES                                       """
##############################################################################################################################

class MultiClassBernoulliNB (object):
    
    def __init__(self):
        
        print("The Multiclass Bernoulli Naive Bayes model (Group 4) has been called.")
    
    def Marginal_Prob (self, y_multiclasslabels):
        
        """ This function calculates the marginal probability of each class P(y=k), i.e. the %frequency of each class in the vector of labels
        INPUT: Vector of labels
        OUTPUT:
            1) List of classes
            2) Absolut frequency of each class
            3) Absolut frequency of each class + 2 needed for the Laplace Smoothing
            4) The list of examples belonging to each class
            5) Vector theta_k of marginal probability of each class with Laplace Smoothing"""
        
        classes , self.class_counter = np.unique(y_multiclasslabels, return_counts=True)
        self.classes = classes.tolist() # List of classes 
        indices = [] # Indices of examples of each class
        for k in range(len(self.classes)):
            indices.append(np.where(y_multiclasslabels == self.classes[k]))
        self.indices = indices
        self.theta_k_vector = self.class_counter / y_multiclasslabels.size # Vector of marginal probability of each class
        self.class_counter_Laplace = (self.class_counter + 2) # Laplace Smoothing
        print("The vector of marginal probabilities of classes was calculated")
        
        
    def Features_Cond_Class_Prob (self, X_binaryfeatures):
        
        """ This function calculates the probabilities of features given the class, i.e. P(x|y=k)
        INPUT: Binary training matrix
        OUTPUT: Matrix Theta_j_k"""
        
        indices = [self.indices[i][0] for i in range(len(self.indices))]
        theta_j_k_matrix = []
        for k in range(len(self.classes)):
            theta_j_k_matrix.append(np.ravel((X_binaryfeatures[indices[k],:].sum(axis=0)))) # Sum the features x=1 of the examples of a given class k
        theta_j_k_matrix = np.array(theta_j_k_matrix)
        theta_j_k_matrix = theta_j_k_matrix + 1 # Laplace Smoothing
        self.theta_j_k_matrix = np.transpose(theta_j_k_matrix)/self.class_counter_Laplace # Calculate P(x|y=k) considering the Laplace smoothing
        print("The matrix of probabilities of features conditional to classes was calculated.")
   
    
    def fit(self, X_binaryfeatures, y_multiclasslabels):
        
        """ This function calls the previous functions to fit the BernoulliNB Model
        INPUT:
            1) Binary training matrix
            2) Vector of labels
        OUTPUT:
            1) Matrix Theta_j_k
            2) Vector Theta_k"""
        
        start_fit = timeit.default_timer()
        
        self.Marginal_Prob( y_multiclasslabels ) # Call function for calculation of theta_k
        self.Features_Cond_Class_Prob ( X_binaryfeatures ) # Call function for calculation of theta_j_k
        
        stop_fit = timeit.default_timer()
        
        print("Bernoulli NB Model was fitted.")
        print("The fitting runtime is %0.4fs"%(stop_fit - start_fit))
        print("\n")
        print("="*100)
        return self.theta_k_vector, self.theta_j_k_matrix
             
    def Class_Cond_Features_Prob (self, X_topredict):
        
        """ Given the matrix we want to make predictions, this function calculates the log-likelihood of each class conditional to data,
        i.e. log(P(y=k|x))
        INPUT: Binary testing matrix to make predictions
        OUTPUT: log-likelihood of each class conditional to data"""
        
        Prob_Class_Feature = np.zeros((X_topredict.shape[0],len(self.classes)))
        for i in range(X_topredict.shape[0]):
            for k in range(len(self.classes)):
                positive_x = np.log(self.theta_j_k_matrix[X_topredict[i,:].nonzero()[1], k ]) # Taking the log(theta_j_k) for features x_j=1 (non-zero values of matrix X)
                temp = np.delete(self.theta_j_k_matrix, X_topredict[i,:].nonzero()[1].tolist(),axis=0)
                negative_x = np.log(1 - temp[:,k]) # Taking the log(1 - theta_j_k) for features x_j = 0 (zero values of matrix X)
                Prob_Class_Feature[i][k] = np.log(self.theta_k_vector[k]) + positive_x.sum() + negative_x.sum() # Calculating matrix log(P(y=|x))
        print("The matrix of probabilities of classes conditional to data was calculated.")
        return Prob_Class_Feature
   
    def predict(self, X_topredict ):
        
        """ This function makes predictions for each example in the testset by calculating the argmax of the log-likelihood of each class conditional to data
        i.e. argmax(log(P(y=k|x)))
        INPUT: Binary testing matrix to make predictions
        OUTPUT: Predictions for each example of the test set"""
        
        start_predict = timeit.default_timer()
        
        Prob = self.Class_Cond_Features_Prob( X_topredict )
        Predictions = []
        for i in range(Prob.shape[0]):
            Predictions.append(self.classes[np.argmax(Prob[i])]) # Taking the argmax(log(P(y|x))) for each example
        self.Predictions = Predictions
        
        stop_predict = timeit.default_timer()
        
        print("Classifications were performed.")
        print("The prediction runtime was %0.4fmin"%((stop_predict - start_predict)/60))
        print("\n")
        print("="*100)
        return self.Predictions


##################################################################################################################################
"""                                           CLASS CUSTOM ENSEMBLE VOTING CLASSIFIER                                         """
##################################################################################################################################

class _BaseVoting(_BaseComposition, TransformerMixin):
    """Base class for voting.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _required_parameters = ['classifier', 'meta_classifier']

    @property
    def named_classifier(self):
        """
        Function to retrieve names of the classification models in the ensemble
        """
        return Bunch(**dict(self.classifier))
    
    @property
    def named_meta_classifier(self):
        """
        Function to retrieve name of the meta-classifier model in the ensemble, but project was abandoned due to time constraint.
        """
        return Bunch(**dict(self.meta_classifier))
        
    def fit_meta(self, X, y):
        """
        Function was intended to generate fitting for a meta-classifier, but was abandoned due to time constraint
        """
        name, metas = zip(*self.meta_classifier)
        self._validate_names(name)

        self.meta_classifier_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(delayed(pf.Fit_Classification_Model)(Classifier=meta, Examples=X, Labels=y, Predict=False, Testing=None)for meta in metas)

        self.named_meta_classifier_ = Bunch()
        for k, e in zip(self.meta_classifier, self.meta_classifier_):
            self.named_meta_classifier_[k[0]] = e
        return self

    def _predict_level_two(self, X):
        """
        Function to make predictions for the meta-classifier... Not finished
        """
        return np.asarray([meta.predict(X) for meta in self.meta_classifier_]).T


class CustomStackVoting(_BaseVoting, ClassifierMixin):
    """ CLASS created to construct a pure ensemble voting classifier without restrictions regarding models' methods.
    Class based on scikit-learn's VotingClassifier. Authors: Sebastian Raschka <se.raschka@gmail.com>, Gilles Louppe <g.louppe@gmail.com>,
    Ramil Nugmanov <stsouko@live.ru> and Mohamed Ali Jamaoui <m.ali.jamaoui@gmail.com>. License: BSD 3 clause
    """
  
    def __init__(self, classifier, voting=None, n_jobs=None, verbose=0):
        """
        Constructor with necessary parameters to instantiate
        """
        self.classifier = classifier
#        self.meta_classifier = meta_classifier
        self.voting = voting
        self.n_jobs = n_jobs
        self.verbose = verbose
#        self.get_meta_features = get_meta_features
        
    def fit(self, X, y):
        """ General fitting function, which calls for a model's fitting function to prevent having to perform label encoding.
        Function also allows parallelization of fitting """
        
        if self.voting not in ('majority') and self.voting != None:
            raise ValueError("Voting must be specified; got (voting=%r)"%self.voting)

        if self.classifier is None or len(self.classifier) == 0:
            raise AttributeError('Invalid "base models" attribute, "classifier"'
                                 ' should be a list of (string, classifier)'
                                 ' tuples')

        names, clfs = zip(*self.classifier)
        self._validate_names(names)

        self.classifier_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(delayed(pf.Fit_Classification_Model)(Classifier=clf, Examples=X, Labels=y, Predict=False, Testing=None)for clf in clfs)
        
        self.named_classifier_ = Bunch()
        for k, e in zip(self.classifier, self.classifier_):
            self.named_classifier_[k[0]] = e
        
        return self

    def predict(self, X):
        """ General prediction function that calls for the model's predict method to prevent heving to back transform the labels
        Function returns list containing the labels with majority of votes 
        ----------------------
        Y* : {list-like vector}, shape (n_sanples) """

        if self.voting == 'majority':
            preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(delayed(pf.Prediction_Classification_Model)(Classifier=clf, Testing=X)for clf in self.classifier_)
            predsT = np.transpose(np.array(preds))
            predsL = predsT.tolist()
            majority = [max(set(lst), key = lst.count) for lst in predsL]
            return majority
        
#        elif self.voting == None and self.meta_classifier != None:
#            name, metas = zip(*self.meta_classifier)
#            predictions = [pf.Prediction_Classification_Model(Classifier=meta, Testing=self.X) for meta in metas]
#            self.predictions_train_meta = self.le_.inverse_transform(maj)
#            preds = np.transpose(self.predictions_train_meta).tolist()
#            preds_total = [' '.join(line) for line in preds]
#            vectorizer = CountVectorizer(decode_error='strict', strip_accents=None, lowercase=False, preprocessor=None, tokenizer=None, stop_words=None,
#                                              ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False)
#            meta_features = vectorizer.fit_transform(preds_total)
#            transformed_y = self.le_.transform(self.y)
#            self.fit_meta(meta_features, transformed_y)
#            predictions_meta = self._predict_meta(X)
#            result = self.le_.inverse_transform(predictions_meta)
#            if(self.get_meta_features):
#                return result, meta_features
#            else:
#            return predictions

    def transform(self, X):
        """Return class labels or probabilities for X for each classifier.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns array-like of shape (n_classifiers, n_samples *
                n_classes), being class probabilities calculated by each
                classifier.
            If `voting='soft' and `flatten_transform=False`:
                array-like of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                array-like of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        check_is_fitted(self, 'classifiers_')
        if self.voting == 'hard':
            return self._predict(X)
