# COMP551 Applied Machine Learning
## Project 2 - NLP Classification

README file to help the understanding, interpretation and use of the python code implemented for Project 2 and submitted with this file

### Code for kaggle competition on reddit comment classification

The entire project was implement using several different python libraries, from data importing and handling to the models used to proceed
with the classification task at hand.

## Installation

The different functions in the code submitted were implemented using a Python 3 base IDE and having all the libraries and dependencies 
updated to their latest version. Two modules require prior installation in order to enable one evaluation function and another to enable
the class StackingCVClassifier (an ensemble meta-classifier model used to generate prediction through base models from Scikit-learn
library). To avoid unexpected errors and to enable the code to work properly, these two libraries' installation are required:

- [Yellowbrick](https://pypi.org/project/yellowbrick/): used for classification report evaluation data visualization
```bash
pip install yellowbrick
```

- [Mlxtend](http://rasbt.github.io/mlxtend/): used to [import](https://pypi.org/project/mlxtend/) the class StackingCVClassifier
```bash
pip install mlxtend
```

_NOTE: The codes found in this repo are from authory of either Luiz Resende Silva, Matheus Faria or Nikhil._

All algorithms were coded by the authors themselves, with exception of the _classification models_, which were all imported
from either Scikit-Learn library or NLTK library (with exception of one of the Bernoulli Naive-Bayes classification model).
