# Machine Learning Algorithms for NLP Classification

### **Authors**: *Resende Silva, Luiz, & Furtado e Faria, Matheus*
### **Date**: *October 2019*
### **Subject**: *COMP551 Applied Machine Learning - Project 02*

### **Overview**
README file to help the understanding, interpretation and use of the python code implemented for NLP classification project. The entire project was implement using several different python libraries, from data importing and handling to the models used to proceed with the classification task at hand.

## INSTALLATION

The different functions in the code submitted were implemented using a Python 3 base IDE and having all the libraries and dependencies updated to their latest version. Two modules require prior installation in order to enable one evaluation function and another to enable the class StackingCVClassifier (an ensemble meta-classifier model used to generate prediction through base models from Scikit-learn library). To avoid unexpected errors and to enable the code to work properly, these two libraries' installation are required:

* [Yellowbrick](https://pypi.org/project/yellowbrick/): used for classification report evaluation data visualization
```bash
pip install yellowbrick
```

* [Mlxtend](http://rasbt.github.io/mlxtend/): used to [import](https://pypi.org/project/mlxtend/) the class ```StackingCVClassifier```
```bash
pip install mlxtend
```

No other module was installed and all the requered class/methods are duly called in the code, releasing the user from the obligation of importing them.

## CODE FILES

The project's code was divided into 5 different .py files to have a cleaner environment and facilitated their use. The files are:

1. *Project_02_Functions.py*: contains the functions designed to be used thoughout the code. Includes dataset importing functions, plotting, calling fit/predict methods from Scikit-learn, text preprocessing, extraction and selection etc. All the functions on it are duly commented and their input parameters are clearly explained (and their names are intuitive). The user does not need to run any of the scripts on it, since other files import its functions. It **must be included in the same directory as the other code files**.

2. *Project_02_UserDefinedClasses.py*: contains the Multinomial Bernoulli Naive-Bayes classification model implemented from scratch and the CustomStackVoting classification model created based on Scikit-learn's [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) to overcome some problems encountered when trying to run the later with models which do not contain the predict_proba method. The user does not need to run any of the scripts on it, since other files import its functions. It **must be included in the same directory as the other code files**.

3. *Project_02_GridSearchModelsParameters.py*: this file contains the code for a grid search set-up used by the author to test and tune the parameters for the different models used. It imports scikit-learn methods and also functions from the file described in item 1 above. *OBS.: the file is not needed to reproduce the results for the predictions submitted - it was used only for the assessment of parameters by the authors - therefore, the file was not included in the submission **code.zip**, it is being mentioned here only for clarification about how the parameters were selected.*
```python
import Project_02_Functions as pf
from Project_02_UserDefinedClasses import CustomStackVoting
from Project_02_UserDefinedClasses import MultiClassBernoulliNB
```
4. *Project_02_TrainingModels.py*: this file contains the code set-up used to assess individual models more thoroughly, where the training dataset for the text classification is imported, splitted in two sets and the models are trained and tested to assess their individual performance. It imports scikit-learn methods and also functions from the files described in items 1 and 2 above. *Use explained below*.

5. *Project_02_Testing_HeldOut.py*: this file contains the code set-up used to actually train the model(s) selected in the entire training dataset and to generate predictions for the testing dataset for the kaggle competition. Both train and test datasets are imported, the models are trained in the entire first and tested in the second. It imports scikit-learn methods and also functions from the files described in items 1 and 2 above. *Use explained below*.

## USAGE

* *Project_02_Functions.py*:
  * ***Functions in this file do not need to be modified***

* *Project_02_UserDefinedClasses.py*:
  * ***Functions and classes in this file do not need to be modified***

* *Project_02_TrainingModels.py*: 
  * First 40 lines all the necessary modules are explicited and imported
  * Lines 45-47: the name of the file containing the dataset is passed and data is imported and copied to a Pandas DataFrame data structure. *OBS.: file name assumes the file is in the same directory. If not the case, the correct path for the training file must be passed to the variable* ```FileTrain```
  * Lines 53-61: the data is analysed regarding its size and histograms describing this data are plotted.
  * Lines 64-67: the preprocessing step is done, where the comments are converted to lowercase, stop words are removed and lemmatization is performed
  * Lines 74-75: the training data is divided in two sets in order to have a set of "unseen" data to assess the accuracy of the model(s) tested. Split is done using Scikit-learn's train_test_split method.
  * Lines 88-91: the vectorizaiton of the comments is done and features are extracted from the training_split dataset. The test_split dataset is transformed to the vocabulary extracted from the training_split set.
  * Lines 94-96: the number of features is reduced by using Feature Selection methods.  *OBS.: tests showed that reducing number of features did not help improve accuracy, therefore these lines were commented out to disable this action of occuring; to enable it, the user just need to uncomment these lines*.
  * Lines 104-140: all the available tested models are instantiated and their parameters are set
  * Lines 145-155: a list containing the models instantiated above is created to be fed to the different ensemble models or to sequentially be used in a loop or called by index.
  * Lines 161-172: a list constaining a selection of the classifiers chosen to be used in the VotingClassifier model is created and the model is instantiated with its required parameters
  * Lines 177-187: a list constaining a selection of the classifiers chosen to be used in the CustomStackVoting model
  * Lines 192-218: a list with the classification models and their respective parameters for the StackingCVClassifier is created, the meta-classifier with its parameters for this model is selected and the ensemble meta-classifier method is instantiated with this list of base models, meta-classifier and needed parameters.
  * Lines 222-224: three flag variables are created to generalize the fitting and prediction steps and enable the user to choose what to run without having to comment/uncomment lines.
  ```python
  SINGLE = True #or False
  ALL = True #or False
  META = True #or False
  ```
  * Lines 230-248:
    * fits and generates predictions for a list of individual models in the Classification_Model function using parallelization to speed-up process, returning a matrix of size (n_samples,n_models) containig the predictions for the different models. To run this line, flags ```SINGLE=True``` and ```ALL=True```
    * fits and generates predictions for a single individual model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=True``` and ```ALL=False```.
    * fits and generates predictions for the CustomStackVoting model in the Classification_Model function, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=False``` and ```META=False```
    * fits and generates predictions for the StackingCVClassifier model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=False``` and ```META=True```
  * Lines 254: a confusion matrix is generated to assess performance of the model trained.

* *Project_02_Testing_HeldOut.py*: 
  * First 35 lines all the necessary modules are explicited and imported
  * Lines 39-43: the names of the files containing the datasets are passed and data is imported and copied to a Pandas DataFrame data structure. *OBS.: file names assume the files are in the same directory. If not the case, the correct path for the training and testing files must be passed to the variables* ```FileTrain``` *and* ```FileTest```*, respectively*
  * Lines 49-64: the preprocessing step is done in both datasets, where the comments are converted to lowercase, stop words are removed and lemmatization is performed
  * Lines 72-75: the vectorizaiton of the comments is done and features are extracted from the training dataset and the test dataset is transformed
  * Lines 83-118: all the available tested models are instantiated and their parameters are set
  * Lines 123-134: a list containing the models instantiated above is created to be fed to ensemble models
  * Lines 139-149: a list constaining a selection of the classifiers chosen to be used in the CustomStackVoting model
  * Lines 154-180: a list with the classification models and their respective parameters for the StackingCVClassifier is created, the meta-classifier with its parameters for this model is selected and the ensemble meta-classifier method is instantiated with this list of base models, meta-classifier and needed parameters.
  * Lines 184-186: three flag variables are created (SINGLE, ALL and META) to generalize the fitting and prediction steps and enable the user to choose what to run without having to comment/uncomment lines.
  * Lines 192-208:
    * fits and generates predictions for a list of individual models in the Classification_Model function using parallelization to speed-up process, returning a matrix of size (n_samples,n_models) containig the predictions for the different models. To run this line, flags ```SINGLE=True``` and ```ALL=True```
    * fits and generates predictions for a single individual model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=True``` and ```ALL=False```.
    * fits and generates predictions for the CustomStackVoting model in the Classification_Model function, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=False``` and ```META=False```
    * fits and generates predictions for the StackingCVClassifier model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags ```SINGLE=False``` and ```META=True```
  * Lines 215-218: the predictions generated by any of the four options of fit/predict above are saved in a Pandas DataFrame structure and a .csv file containing the respective IDs for the predictions and the predictions is generated in order to be submitted to the kaggle competition.

### RESULTS REPRODUCTION AND FEEDBACK

The files described above were submitted containing the correct parameters in order to enable the results' reproduction for both the last training made and the prediction on the test set , i.e. *by only running the scripts the results generated would be the ones submitted*. The model used in the kaggle submission was the StackingCVClassifier ran using the parameters below:
```python
ListModelMetaClassifier = [MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True) #0
                    ,BernoulliNB(alpha=0.01, class_prior=None, fit_prior=True) #1
                    ,RandomForestClassifier(n_estimators=2500, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                          max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=-1, oob_score=False,
                          random_state=None, verbose=0, warm_start=False) #2
                    ,ExtraTreesClassifier(n_estimators=1500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                         min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                         min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                         warm_start=False, class_weight=None) #3
                    ,GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=250, subsample=1.0,
                                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=2, 
                                                min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                                min_impurity_split=None, init=None, random_state=None, max_features=None,
                                                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
                                                validation_fraction=0.1, n_iter_no_change=None, tol=0.0001) #4
                    ,AdaBoostClassifier(base_estimator=kSVC, n_estimators=10, learning_rate=1.0, algorithm='SAMME', random_state=None)#5
                   ]

Meta_Estimator = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=10, fit_intercept=True, intercept_scaling=1,
                              class_weight=None, random_state=None, solver='saga', max_iter=1500, multi_class='auto', verbose=0,   
                              warm_start=False, n_jobs=-1,l1_ratio=0.75)
                              
MetaClass = StackingCVClassifier(classifiers=ListModelMetaClassifier, meta_classifier=Meta_Estimator, use_probas=True,
                                 drop_last_proba=False, cv=3, shuffle=False, random_state=None, stratify=True, verbose=2,
                                 use_features_in_secondary=False,  store_train_meta_features=False, use_clones=True, n_jobs=-1)
```
**OBS.1:** other models were tested, but this meta-classifier, to the knowledge of the author, has restrictions about which models can be fed to it. Since it uses the ```predict_proba``` method, models which do not present this method cannot be used. It seems to be a glitch in the class, since the author tested different models using the parameter ```use_probas=False```, however, even with this parameters set to ```False```, it throws an error.

**OBS.2:** the base models fed to the StackingCVClassifier must have the parameter ```verbose=0```, or else the script will throw an exception 
```python
 AttributeError: 'NoneType' object has no attribute 'write'
```
This error seems to be related to the verbose of the class itself, which aparently cannot handle the verbose of the base models and meta-estimator.

*Note:* The user is invited to read the comments and descriptions of the functions in *Project_02_Functions.py* and in the classification models and see if better results can be achieved. If so, one can reach the authors through ```luiz.resendesilva@mail.mcgill.ca```.

## FUTURE UPDATES

The Multinomial Bernoulli Naive-Bayes classification model implemented from scratch was prove to take a longer time when generating the predictions. A possible future upgrade would be the implementation of the parallelization of some internal functions iterating through very sparce matrices and replacing some loops by vector operations.
Another intended update is the implementation of furhter functions in the CustomStackVoting to enable the use of meta classification by feeding the predicitions of ```k``` base estimators to a meta-estimator that would be trained upon this data and learn to make predictions based on the performance of the ```k``` base estimators, as in StackingCVClassifier.
Such updates were not implement in the classes submitted due to time restrictions.

## AUTHORS AND ACKNOWLEDGMENTS

### AUTHORSHIP
All the scripts presented in the files *Project_02_Functions.py*, *Project_02_GridSearchModelsParameters.py*, *Project_02_TrainingModels.py* and *Project_02_Testing_HeldOut.py* (with the exception of the imported modules) were coded/implementd by Luiz Resende Silva. The scripts presented in *Project_02_UserDefinedClasses.py* file were coded by both authors.

All classification, preprocessing and utilities modules were imported from either Scikit-Learn, NLTK and Mlxtend, with exception of the MultiClassBernoulliNB classification model implemented from scratch. All other modules were general libraries used in handling data and support implemented functions.

### LICENSE
All files made available in this repository follow the licensing guidelines defined in the file **LICENSE.md**.
