# COMP551 APPLIED MACHINE LEARNING - PROJECT 2
## GROUP 04 - Luiz Resende Silva, Matheus Faria and Nikhil Krishna
### NLP Classification

README file to help the understanding, interpretation and use of the python code implemented for Project 2 and submitted with this file

### Code for kaggle competition on reddit comment classification

The entire project was implement using several different python libraries, from data importing and handling to the models used to proceed with the classification task at hand.

## INSTALLATION

The different functions in the code submitted were implemented using a Python 3 base IDE and having all the libraries and dependencies updated to their latest version. Two modules require prior installation in order to enable one evaluation function and another to enable the class StackingCVClassifier (an ensemble meta-classifier model used to generate prediction through base models from Scikit-learn library). To avoid unexpected errors and to enable the code to work properly, these two libraries' installation are required:

* [Yellowbrick](https://pypi.org/project/yellowbrick/): used for classification report evaluation data visualization
```bash
pip install yellowbrick
```

* [Mlxtend](http://rasbt.github.io/mlxtend/): used to [import](https://pypi.org/project/mlxtend/) the class StackingCVClassifier
```bash
pip install mlxtend
```

No other module was installed and all the requered class/methods are duly called in the code, releasing the user from the obligation of importing them.

## CODE FILES

The project's code was divided into 5 different .py files to have a cleaner environment and facilitated their use. The files are:

1. *Project_02_Functions.py*: contains the functions designed to be used thoughout the code. Includes dataset importing functions, plotting, calling fit/predict methods from Scikit-learn, text preprocessing, extraction and selection etc. All the functions on it are duly commented and their input parameters are clearly explained (and their names are intuitive). The user does not need to run any of the scripts on it, since other files import its functions. It **must be included in the same directory as the other code files**.

2. *Project_02_UserDefinedClasses.py*: contains the Multinomial Bernoulli Naive-Bayes classification model implemented from scratch and the CustomStackVoting classification model created based on Scikit-learn's [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) to overcome some problems encountered when trying to run the later with models which do not contain the predict_proba method. The user does not need to run any of the scripts on it, since other files import its functions. It **must be included in the same directory as the other code files**.

3. *Project_02_GridSearchModelsParameters.py*: this file contains the code for a grid search set-up used by the author to test and tune the parameters for the different models used. It imports scikit-learn methods and also functions from the file described in item 1 above.
*OBS.: the file is not needed to reproduce the results for the predictions submitted - it was used only for the assessment of parameters by the authors - therefore, the file was not included in the submission **code.zip**, it is being mentioned here only for clarification about how the parameters were selected.*

4. *Project_02_TrainingModels.py*: this file contains the code set-up used to assess individual models more thoroughly, where the training dataset for the text classification is imported, splitted in two sets and the models are trained and tested to assess their individual performance. It imports scikit-learn methods and also functions from the files described in items 1 and 2 above. *Use explained below*.

5. *Project_02_Testing_HeldOut.py*: this file contains the code set-up used to actually train the model(s) selected in the entire training dataset and to generate predictions for the testing dataset for the kaggle competition. Both train and test datasets are imported, the models are trained in the entire first and tested in the second. It imports scikit-learn methods and also functions from the files described in items 1 and 2 above. *Use explained below*.

## USAGE

* *Project_02_TrainingModels.py*: 

* *Project_02_Testing_HeldOut.py*: 
  * First 64 lines all the necessary modules are explicited and imported
  * Lines 65-72: the names of the files containing the datasets are passed and data is imported and copied to a Pandas DataFrame data structure. *OBS.: file names assume the files are in the same directory. If not the case, the correct path for the training and testing files must be passed to the variables **FileTrain** and **FileTest**, respectively*
  * Lines 74-83: the preprocessing step is done in both datasets, where the comments are converted to lowercase, stop words are removed and lemmatization is performed
  * Lines 91-103: the vectorizaiton of the comments is done and features are extracted from the training dataset and the test dataset is transformed
  * Lines 105-143: all the available tested models are instantiated and their parameters are set
  * Lines 145-159: a list containing the models instantiated above is created to be fed to ensemble models
  * Lines 161-174: a list constaining a selection of the classifiers chosen to be used in the CustomStackVoting model
  * Lines 176-204: a list with the classification models and their respective parameters for the StackingCVClassifier is created, the meta-classifier with its parameters for this model is selected and the ensemble meta-classifier method is instantiated with this list of base models, meta-classifier and needed parameters.
  * Lines 209-211: three flag variables are created (SINGLE, ALL and META) to generalize the fitting and prediction steps and enable the user to choose what to run without having to comment/uncomment lines.
  * Lines 219-220: fits and generates predictions for a list of individual models in the Classification_Model function using parallelization to speed-up process, returning a matrix of size (n_samples,n_models) containig the predictions for the different models. To run this line, flags SINGLE=True and ALL=True
  * Lines 223-225: fits and generates predictions for a single individual model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags SINGLE=True and ALL=False.
  * Lines 227-229: fits and generates predictions for the CustomStackVoting model in the Classification_Model function, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags SINGLE=False and META=False
  * Lines 231-233: fits and generates predictions for the StackingCVClassifier model, returning a vector of size (n_samples,1) containig the predictions. To run this line, flags SINGLE=False and META=True
  
  
  

*NOTE: The codes found in this repo are from authory of either Luiz Resende Silva, Matheus Faria or Nikhil.*

All algorithms were coded by the authors themselves, with exception of the _classification models_, which were all imported
from either Scikit-Learn library or NLTK library (with exception of one of the Bernoulli Naive-Bayes classification model).
