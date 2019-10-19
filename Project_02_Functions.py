# -*- coding: utf-8 -*-
"""
@author: Luiz Resende Silva - ID 260852243
"""
##################################################################################################################################
'''                                             IMPORTING GENERAL LIBRARIES                                                   '''
##################################################################################################################################
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time
import timeit
import re
from bs4 import BeautifulSoup
import unicodedata
from scipy import sparse
from scipy.sparse import hstack
##################################################################################################################################
'''                                      IMPORTING SCIKIT-LEARN METHODS AND MODULES                                          '''
##################################################################################################################################
##################################################################
'''                 FEATURE EXTRACTION                       '''
##################################################################
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import cross_val_score
from sklearn import metrics
##################################################################
'''      STACKING CROSS-VALIDATION META-CLASSIFIE            '''
##################################################################
from mlxtend.classifier import StackingCVClassifier
##################################################################
'''                     NLTK LIBRARY MODULES                 '''
##################################################################
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
##################################################################
'''     PY FILE CONTAINING FUNCTIONS & CLASSES BUILT       '''
##################################################################
#from Project_02_UserDefinedClasses import CustomStackVoting
##################################################################################################################################
'''                                 GENERAL FUNCTIONS TO READ, WRITE AND HABDLE DATA                                        '''
##################################################################################################################################

def Read_File_DF(File_Name, separation=",", head=None, replace=[], drop=False):
    """ Function created to read dataset files and import information as a Pandas DataFrame
    INPUT: Dataset file name, character delimiter in the file (default ','), flag for file header (default 0 - no header head=None),
            list of strings to find possible malformations (default empty) and flag to drop or not lines/columns
            containing such values (default False)"""
    try:
        separation = separation.lower()
        if(separation == "space" or separation == "tab"):
            separation = "\t"
        Raw_Data_Set = pd.read_csv(File_Name, delimiter=separation, header=head, na_values=replace)
        RawRowsColumns = Raw_Data_Set.shape
        if(replace != None):
            Missing = Raw_Data_Set.isnull().sum().sum()
            print("Total number of missing/anomalous 'entries' in the data set: ",Missing)
            if(drop == True):
                Raw_Data_Set.dropna(axis=0, how='any', inplace=True)
                CleanRowsColumns = Raw_Data_Set.shape
                print("Number of examples with missing values deleted from data set: ",(RawRowsColumns[0]-CleanRowsColumns[0]))
        return Raw_Data_Set
    except:
        print("READ_FILE_ERROR\n")

def Write_File_DF(Data_Set, File_Name="Predictions", separation=",", head=True, ind=False, dec='.'):
    """ Function created to write a Pandas DataFrame containing prediciton made by classifiers to submit to competition
    INPUT: DataFrame with IDs and predictions, file name (default 'Predictions'), character delimiter in the file (default ','), 
            flag to include file header (default True), flag to include column of indices (default False)
            and character for decimals (default '.') """
    try:
        separation = separation.lower()
        if(separation == "space" or separation == "tab"):
            separation = "\t"
        timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
        name = timestr+File_Name+".csv"
        Data_Set.to_csv(path_or_buf=name, sep=separation, na_rep='', float_format=None, columns=None, header=head, index=ind,
                        index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
                        line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal=dec)
        return print("Predictions for classes exported as csv file")
    except:
        print("WRITE_FILE_ERROR\n")

def To_PD_DF(Data_Set):
    """ Function created to convert NumPy array to Pandas DataFrame 
    INPUT: NumPy array/matrix containing data """
    try:
        Data_Set_DF = pd.DataFrame(Data_Set, index = (list(range(0,Data_Set.shape[0]))), columns = (list(range(0,Data_Set.shape[1]))))
        return Data_Set_DF
    except:
        print("DATAFRAME_CONVERT_ERROR\n")
    
def To_NP_Array(Data_Set):
    """ Function created to convert Pandas DataFrame to NumPy array/matrix 
    INPUT: Pandas DataFrame containing data """
    try:
        Data_Set_NP = Data_Set.to_numpy(copy = True)
        return Data_Set_NP
    except:
        print("NP_CONVERT_ERROR\n")

##################################################################################################################################
'''                                            GENERAL MISCELANEUOUS FUNCTIONS                                           '''
##################################################################################################################################

def Data_Stats(Data, QQ_DD=True, show=False):
    """ Function created to calculate and show/save some basic statistics and correlation about the dataser
    INPUT: DataFrame dataset, flag for quartiles or deciles (default True=quartiles) and flag for printing information to screen (default False) """
    try:
        Data_Set = pd.DataFrame(Data, index = (list(range(0,Data.shape[0]))), columns = (list(range(0,Data.shape[1]))))
        if(QQ_DD == True):           
            quantiles = [0.00, 0.25, 0.50, 0.75] #Calculating quartiles
        else:
            quantiles = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00] #Calculating quartiles

        Describ = Data_Set.describe(percentiles = quantiles) #Data set general stats description
        Correlation = Data_Set.corr('spearman') #Computing pairwise feature correlation through Spearman rank correlation
        name = ("GeneralStats.xlsx")
        with pd.ExcelWriter(name) as writer: #Outputting Excel file with statistics
            Describ.to_excel(writer, sheet_name='Data_Description')
            Correlation.to_excel(writer, sheet_name='Column_Correlation')
        if(show == True):
            print(Data_Set)
            print(Describ) #Printing statistics to screen
            print(Correlation) #Printing statistics to screen
    except:
        print("STATS_FUNCTION_ERROR\n")

##################################################################################################################################
'''                                            PLOTTING/GRAPHICS FUNCTIONS                                              '''
##################################################################################################################################

def Count_Plot(Column=None, Dataset=None, Cond=None, TitleName="MyPlot", Color=None, save=False, Size=(25,20)):
    """ Function imports method to analyse the DataFrame or DataSeries distribution
    INPUT: Column containing the classes, dataset, hue to a third value (default None), title, color to be used (default Rainbow),
			flag to save or not the figure (default False) and tuple for the figure size (default (25,20)) """
    try:
        sb.set(style="whitegrid", color_codes=True)
        sb.set(rc={"figure.figsize":Size})
        if(Column!=None):
            ax = sb.countplot(x=Column,data=Dataset, hue=Cond, color=Color) #Create a countplot and define hue if Cond!=None
        else:
            ax = sb.countplot(Dataset, hue=Cond, color=Color)
        ax.set_title(TitleName, fontsize = 32)
        plt.show()
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = ax.get_figure()
            fig.savefig(timestr+TitleName+".png")
    except:
        print("COUNTPLOT_GENERATION_ERROR\n")

def ClassReport_Graph(Classif, Data_train, Target_train, Data_test, Target_test, Class, ModelName='Classifier', Accur=False, Predict=None):
    """ Function imports method to report and analyse predictions from different scikit-learn model implementations
    INPUT: training examples' features, training examples' outputs, testing examples' features, testing examples' outputs
            and list with the names of the classes """
    try:
        from yellowbrick.classifier import ClassificationReport
        
        if(Accur==True):
            print((ModelName+" accuracy: %0.4f")%(metrics.accuracy_score(Target_test, Predict, normalize=True)))
        
        view_graph = ClassificationReport(Classif, classes=Class, size=(900, 720)) #Object for classification model and visualization
        view_graph.fit(Data_train, Target_train) # Fit the training data to the visualizer
        view_graph.score(Data_test, Target_test) # Evaluate the model on the test data
        graph = view_graph.poof() # Draw/show/poof the data
        return graph
    except:
        print("CLASSIFICATION-REPORT_ERROR\n")

def Learn_Perform(DataF, LabelX='Classifiers', LabelY1='Accuracy', LabelY2='Run Time', TitleName="Resulting Scores", Size=(16,12), save=False):        
    """ Function designed to plot the performance of the best results and parameters for the different learning model fed to
		to the Pipeline in the GridSearch function 
    INPUT: DataFrame containing the results (accuracies and times), name of column containing learner names, name of accuracy axis,
		name for the running time axis, title for the plot, size of the plot and flag for saving or not (default False)
			and flag to save or not the figure (default False)"""
    try:
        DF = pd.melt(DataF, id_vars=LabelX, var_name='Variables', value_name='value_numbers')
        fig, ax1 = plt.subplots(figsize=Size)
        graph = sb.barplot(x=LabelX, y='value_numbers', hue='Variables', data=DF, ax=ax1)
        ax2 = ax1.twinx()
        ax1.set_title(TitleName, fontsize = 24)
        ax1.set_xlabel(LabelX, fontsize=18)
        ax1.set_ylabel((LabelY1+" (%)"), fontsize=18)
        ax1.set_ylim(0.0,100.0)
        ax2.set_ylabel((LabelY2+" (s)"), fontsize=18)
        ax2.set_ylim(0,DataF[LabelY2].max())
        plt.setp(ax1.get_legend().get_texts(), fontsize='16') # for legend text
        plt.show()
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            graph = ax1.get_figure()
            graph.savefig(timestr+TitleName+".png")
    except:
        print("LEARNER-PERFORM_ERROR\n")

def Get_ConfusionMatrix(TrueLabels, PredictedLabels, Classes, Normal=False, Title='Confusion matrix', ColorMap='rainbow',
                        FigSize=(30,30), save=False):
    """ Function designed to plot the confusion matrix of the predicted labels versus the true leabels 
    INPUT: vector containing the actual true labels, vector containing the predicted labels, flag for normalizing the data (default False),
            name of the title for the graph, color map (default winter) and flag to save or not the figure (default False).
    OUTPUT: function returns a matrix containing the confusion matrix values """
#   Colormap reference -> https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    try:   
        ConfMatrix = metrics.confusion_matrix(TrueLabels, PredictedLabels) #Calculating confusion matrix
    
        if(Normal==True):
            ConfMatrix = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

        ConfMatrix_DF = pd.DataFrame(data=ConfMatrix, index=Classes, columns=Classes)                     
        fig, ax = plt.subplots(figsize=FigSize)
        sb.heatmap(ConfMatrix_DF, annot=True, cmap=ColorMap)
        ax.set_title(Title, fontsize=26)
        ax.set_xlabel('Predicted labels', fontsize = 20)
        ax.set_ylabel('True labels', fontsize = 20)
        ax.set_ylim(len(ConfMatrix)+0.25, -0.25)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
    
        if(save==True):
            timestr = time.strftime("%y-%m-%d_%Hh%Mm%Ss_")
            fig = ax.get_figure()
            fig.savefig(timestr+Title+".png")
    
        return ConfMatrix_DF
    except:
       print("CONFUSION-MATRIX_ERROR\n") 

##################################################################################################################################
'''                                 CLASS AND FUNCTIONS TO CALL NLTK METHODS AND MODULES                                     '''
##################################################################################################################################

""" IN THE FIRST RUN, THESE LINES MUST BE UNCOMMENTED IN ORDER TO DOWNLOAD THE PACKAGES """
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

Music_List = ['album','band','bass','cd','dance','deezer','dj','drums','guitar','jazz','lyric','metal','mp3','music','musician','playlist','pop','radio','sing','singer','song','sound','spotify','tracklist','vinyl']

GO_Lsit = ['c4','c9','counter','cs','csgo','ct','cts','ghz','globaloffensive','strike']

def Document_Preprocess(Doc_Comment, LowerCase=True, RemoveHTML=True, StripAccent=True, Accented='ascii', StripCharSpec=True, RemoveStop=True, StopWords=(stopwords.words('english')), DoLemma=True, DoSplit=False):
    """ Function designed to perform preprocessing of the text features. Function gives possibility of removing stop words and performing
    lemmatization of the documments in the corpus
    INPUT: decument/sentence to be processed (string), flag for removing stop words (default True), lsit of stop words to use (default
            NLTK english stop word list), flag to perform lemmatization or not (NLTK module) and flaag to return list of tokens (default False)
    OUTPUT: function returns either a string or a list depending on the flag DoSplit """
    if(LowerCase==True):
        Doc_Comment = Doc_Comment.lower() #Putting everything in lower case
    if(RemoveHTML==True):
        beatsoup = BeautifulSoup(Doc_Comment, "html.parser")
        Doc_Comment = beatsoup.get_text()
    if(RemoveStop==True):
        if(LowerCase):            
            Doc_Comment = [token for token in Doc_Comment.split() if token not in StopWords] #Tokenizing and removing stop words if flag True
        else:
            Doc_Split = Doc_Comment.split()
            Doc_Comment = [token for token in Doc_Split if token not in StopWords]
        Doc_Comment = ' '.join(Doc_Comment) #Rejoining sentence    
    if(StripAccent==True):
        Doc_Comment = unicodedata.normalize('NFKD', Doc_Comment).encode(Accented, 'ignore').decode('utf-8', 'ignore')
    if(StripCharSpec==True):
        Doc_Comment = re.sub(r'[^a-zA-Z0-9+]', ' ', Doc_Comment, re.I|re.A) #Getting rid-off of special characters and punctuation
        Doc_Comment = re.sub(r'[\r|\n|\r\n]+', ' ', Doc_Comment) #Getting rid-off of extra new lines
        Doc_Comment = Doc_Comment.strip()
    if(DoLemma==True):
        wnl = WordNetLemmatizer() #Lemmatizing if flag true
        Doc_Split = Doc_Comment.split()
        Doc_Comment = [wnl.lemmatize(token) for token in Doc_Split]
        Doc_Comment = ' '.join(Doc_Comment) #Rejoining sentence
    if(DoSplit==True):
        Doc_Comment = Doc_Comment.split()
    return Doc_Comment

def Check_Lists(LsitCheck, ListElem, Weight=10):
    """ Function designed to create a new feature column by analysing the presence of certian words from a given list in the comments
    INPUT: text splited in words (list of string) to be checked, list containing words to be found (list of string) and respective weight
            to be returned (default 10)
            NLTK english stop word list) and flag to perform lemmatization or not (NLTK module) """
    Presence =  any(token in LsitCheck for token in ListElem)
    if Presence:
        return (1*Weight)    
    else:
        return (0*Weight)

def StopW_Punct():
    """ Function creates a list containing strings of punctuation to use as stop words """
    punctList = ["!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","{","|","}","~","^","_","]","`"]
    return punctList

def StopW_NLTK(DicLan='english'):
    """ Function returns a list containing the NLTK library stop words
    INPUT: dictionary language to follow (default 'english') """
    try:
        nltkStopWordList = stopwords.words(DicLan)
        if(DicLan=='english'):
            nltkStopWordList.append("i'm")
        return nltkStopWordList
    except:
        print("NLTK_STOPWORDS_ERROR\n")

class LemmatizerTokens(object): #Lemmatazing class retrieved from Scikit-Lear API to be used in the CountVectorizer and TfidfVectorizer extractors
    """ Class created to return lemmatizing function from NLTK library """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

##################################################################################################################################
'''         FUNCTION TO PERFORM GRID SEARCH IN THE FEATURE EXTRACTION/SELECTION PARAMETERS AND DIMENSION                     '''
##################################################################################################################################        
        
def ClassificationModelGrid(Examples, Labels, Classifier, Name, GridParameters=None, Folds=2, Normal=False, UseSelec=False,
                            Ensemble=False, GetFitted=False):
    """ Function designed to receive list of classification learner and a list of parameters, construct a pipeline with them and access a grid search  
    INPUT: Structure containing training examples' features and another containing trainig examples' labels, classifier,
            name given to the classifier, parameters for the grid search (default None), number of folds for validation (default 2), flag
            for normalizing the data (default False), flag for use feature selection (default False), flag for feeding ensemble voting
            classifier to the grid search (default False) and flag for returning fitted model to make predictions (default False)
    OUTPUT: the function returns the name of the classifier trained, it's average accuracy in kFold and the total time spent in the kFold,
            and also the fitted model if UseSelec=True """
    
    print('-' * 100)
    print("Training model: ")
    print(Classifier)
    
    """ DIFFERENT OPTIONS FOR USE OF FEATURE EXTRACTORS, SELECTION, NORMALIZATION AND CLASSIFIER IN PIPELINE """
    print("Creating pipeline...")
    if(UseSelec==True):
        if(Normal==True):   
            pipeline = Pipeline([('vectorizer', TfidfVectorizer()),
                               ('normalizer', Normalizer()),
                               ('selector', SelectKBest()),
                               ('classifier', Classifier)])
        elif(Normal==False):
            pipeline = Pipeline([('vectorizer', TfidfVectorizer()),
                               ('selector', SelectKBest()),
                               ('classifier', Classifier)])
    elif(UseSelec==False):
        if(Normal==True):   
            pipeline = Pipeline([('vectorizer', TfidfVectorizer()),
                               ('normalizer', Normalizer()),
                               ('classifier', Classifier)])
        elif(Normal==False):
            pipeline = Pipeline([('vectorizer', TfidfVectorizer()),
                               ('classifier', Classifier)])
    
    print("... pipeline created.")
  
    """ CREATING GRIDSEARCH WITH CHOSEN PIPELINE ABOVE """
    if(Ensemble==False):
        gridSearchClf = GridSearchCV(pipeline, param_grid=GridParameters, cv=Folds, n_jobs=-1, iid='warn', refit=True, verbose=1,
                                  error_score='raise-deprecating', return_train_score=False)
    elif(Ensemble==True):
        vectorizer = TfidfVectorizer(decode_error='strict', strip_accents='unicode', lowercase=True, preprocessor=None,
                                                  tokenizer=None, analyzer='word', stop_words='english',
                                                  ngram_range=(1, 1), max_df=1.0, max_features=None, vocabulary=None,
                                                  binary=False, norm='l1', use_idf=True, smooth_idf=True, sublinear_tf=True)
        Examples = vectorizer.fit_transform(Examples)
        Examples = vectorizer.transform(Labels)        
        gridSearchClf = GridSearchCV(estimator=Classifier, param_grid=GridParameters, cv=Folds, n_jobs=-1, iid='warn', refit=True, verbose=1,
                                  error_score='raise-deprecating', return_train_score=False)
    
    tStart = timeit.default_timer() #Starting clock to measure time in the kFold
    gridSearchClf.fit(Examples,Labels) #Running grid search
    runingTime = (timeit.default_timer() - tStart) #Stopping clock and getting time spent
    print("Model fitted and k-Fold Cross Validation done in  %0.4fs"%runingTime)
    
    Accuracy = gridSearchClf.best_score_  #Retrieving best model configuration accuracy
    print("Best accuracy of: %0.4f"%gridSearchClf.best_score_)
    
    print("Best set of parameters:")
    BestParameters = gridSearchClf.best_params_ #Retrieving best model configuration parameters
    print(BestParameters)
    
    if(GetFitted==False):
        return Name, Accuracy, runingTime
    elif(GetFitted==True):
        return Name, Accuracy, runingTime, gridSearchClf

##################################################################################################################################
'''                                 FUNCTION TO CALL SCIKIT-LEARN METHODS AND MODULES                                        '''
##################################################################################################################################

def ExtractVectorizer(DataTrain, DataTest, CountVecXTFIDF=True, Accents=None, Token=None, Stop=None, nGram=None, Binar=False, Regular='l2',
                      Normal=False, SubLinear=False, MinDf=1, AdditFeat=False, FeatTrainToAdd=None, FeatTestToAdd=None, show=False):
    """ Function imports method CountVectorizer and TfidfVectorizer from scikit-learn to extract text features
    INPUT: training examples' features, testing examples' features, flag for choosing extractor (default True -> CountVectorizer),
            string for strip accents (default None), tokenizer method (default none), list containing set of stop_words (default None),
            the tupple fortusing n-grams (default None -> (1,1)), flag to binarize CountVectorizer, regularization method for Tfidf (default 'l2'),
            flag to normalize or not the extracted features (default False), flag for sublinear tf (default False), flag for additional
            features which are not to be fitted (default False), training additional features, testing additional features and
            flag for showing resulting vectors (default False)
    OUTPUT: Vector of fitted training features and vector of transformed test features """
    try:
#        Default arguments CountVectorizer -> (encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
#                                              preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b',
#                                              ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None,
#                                              vocabulary=None, binary=False)
#        Default arguments TfidfVectorizer -> (encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
#                                              preprocessor=None, tokenizer=None, analyzer='word', stop_words=None,
#                                              token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1,
#                                              max_features=None, vocabulary=None, binary=False, norm='l2', use_idf=True,
#                                              smooth_idf=True, sublinear_tf=False)
        if(nGram==None):
            nGram = (1,1)
        
        if(CountVecXTFIDF==True):
            vectorizing = CountVectorizer(strip_accents=Accents, tokenizer=Token, stop_words=Stop, ngram_range=nGram, binary=Binar, min_df=MinDf)
            print("Vectorizer selected:",vectorizing)
        elif(CountVecXTFIDF==False):
            vectorizing = TfidfVectorizer(strip_accents=Accents, tokenizer=Token, stop_words=Stop, ngram_range=nGram, min_df=MinDf,
                                          binary=Binar, norm=Regular, sublinear_tf=SubLinear)
            print("Vectorizer selected:",vectorizing)
        
        vector_training = vectorizing.fit_transform(DataTrain) #Vectorizing vocabulary for training set and fitting        
        vector_testing = vectorizing.transform(DataTest) #Vectorizing vocabulary for test set and transforming - only transforming, or else we would train in the test data also
        
        if(AdditFeat==True):
            sparse_matrix1 = sparse.csr_matrix(FeatTrainToAdd)
            sparse_matrix2 = sparse.csr_matrix(FeatTestToAdd)
            vector_training = hstack((vector_training, sparse_matrix1))
            vector_testing = hstack((vector_testing, sparse_matrix2))
            print("All features assembled.\n")
        
        if(Normal==True):
            vector_training = normalize(vector_training) #Normalizing vector for training set        
            vector_testing = normalize(vector_testing) #Normalizing vector for training set
            print("Data normalized.\n")
        
        if(show==True):
            print("Mapping of terms:",vectorizing.vocabulary_) #Vocabulaty from the vectorization
            print("List of stop words used:",vectorizing.stop_words_) #Terms ignored in the vectorization
            if hasattr(vectorizing, 'idf_'):
                print("Inverse document frequency:",vectorizing.idf_) #Inverse document frequency from the vectorization
        return vector_training, vector_testing
    except:
        print("VECTORIZER_ERROR\n")

def SelectionFeature(DataTrain, TargetTrain=None, DataTest=None, Testing=False, SelectKxSelectP=True, kPick='all', Percent=100.0,
                     Chi2xMutual=True, MutualN=3, AdditFeat=False, FeatTrainToAdd=None, FeatTestToAdd=None, Normal=False, show=False):
    """ Function imports methods of feature selection from scikit-learn
    INPUT: training examples' features, training examples targets, testing examples' features, flag for validating or generating
            predictions to submit (default False), flag for choosing selection model (default True -> SelectKBest), number of features
            to hold in SelectKBest, percentage of features to retain in SelectPercentile, flag for score function (default True -> chi2)
            number of neighbors in mutual_info_classif (default 3), flag for additional features which are not to be fitted
            (default False), training additional features, testing additional features and flag for showing resulting vectors (default False)
    OUTPUT: Vector of fitted training features and vector of transformed test features """
    try:
#        Default arguments SelectKBest -> (score_func=<function f_classif>, k=10)
#        
#        Default arguments SelectPercentile -> (score_func=<function f_classif>, percentile=10)
#        
#        Default arguments chi2 -> (X, y)
#        
#        Default arguments mutual_info_classif -> (X, y, discrete_features=’auto’, n_neighbors=3, copy=True, random_state=None)
              
        if(SelectKxSelectP==True and Chi2xMutual==True):
            selector = SelectKBest(chi2, k=kPick)
            print("Feature selection chosen:",selector)
            if(Testing==False):
                vector_selec_train = selector.fit_transform(DataTrain, TargetTrain)
                vector_selec_test = selector.transform(DataTest)
            elif(Testing==True):
                vector_selec_train = selector.transform(DataTrain)
        elif(SelectKxSelectP==True and Chi2xMutual==False):
            selector = SelectKBest(mutual_info_classif(n_neighbors=MutualN), k=kPick)
            print("Feature selection chosen:",selector)
            if(Testing==False):
                vector_selec_train = selector.fit_transform(DataTrain, TargetTrain)
                vector_selec_test = selector.transform(DataTest)
            elif(Testing==True):
                vector_selec_train = selector.transform(DataTrain)
                
        elif(SelectKxSelectP==False and Chi2xMutual==False):
            selector = SelectPercentile(mutual_info_classif(n_neighbors=MutualN), percentile=Percent)
            print("Feature selection chosen:",selector)
            if(Testing==False):
                vector_selec_train = selector.fit_transform(DataTrain, TargetTrain)
                vector_selec_test = selector.transform(DataTest)
            elif(Testing==True):
                vector_selec_train = selector.transform(DataTrain)
        elif(SelectKxSelectP==False and Chi2xMutual==True):
            selector = SelectPercentile(chi2, percentile=Percent)
            print("Feature selection chosen:",selector)
            if(Testing==False):
                vector_selec_train = selector.fit_transform(DataTrain, TargetTrain)
                vector_selec_test = selector.transform(DataTest)
            elif(Testing==True):
                vector_selec_train = selector.transform(DataTrain)        
            
        if(AdditFeat==True):
            if(Testing==False):
                sparse_matrix1 = sparse.csr_matrix(FeatTrainToAdd)
                sparse_matrix2 = sparse.csr_matrix(FeatTestToAdd)
                vector_selec_train = hstack((vector_selec_train, sparse_matrix1))
                vector_selec_test = hstack((vector_selec_test, sparse_matrix2))
            elif(Testing==True):
                sparse_matrix1 = sparse.csr_matrix(FeatTrainToAdd)
                vector_selec_train = hstack((vector_selec_train, sparse_matrix1))
            print("All features assembled.\n")
        
        if(Normal==True):
            vector_selec_train = normalize(vector_selec_train) #Normalizing vector for training set        
            vector_selec_test = normalize(vector_selec_test) #Normalizing vector for training set
            print("Data normalized.\n")        
        
        if(show==True):
            print("Mapping of terms:",selector.vocabulary_) #Vocabulaty from the vectorization
            print("List of stop words used:",selector.stop_words_) #Terms ignored in the vectorization
            if hasattr(selector, 'idf_'):
                print("Inverse document frequency:",selector.idf_) #Inverse document frequency from the vectorization
        
        if(Testing==False):
            return vector_selec_train, vector_selec_test
        else:
            return vector_selec_train
    except:
        print("VECTORIZER_ERROR\n")

def Classification_Model(data_training, target_training, data_testing, Classifier, target_testing=None, ModelName="Classifier", accur=False,
                         grph=False, setClass=None, show=False):
    """ Function created to receive a classification model from Scikit-Learn library and perform the fittinf and predicting steps.
    INPUT: training examples' features, training examples' outputs, testing examples' features, testing examples' outputs if performing held out
            validation (default None), classifier name, flag for printing accuracy in case of validation (default False), flag for printing
            classification reports (default grph=False) from ClassReport_Graph funciton, list with the names of the classes (if grph=True) 
            and flag fo showing classifier's attributes (default False) """
#    try:
    print("Classifier selected: ", Classifier)
    print("-"*100)
    Classifier.fit(data_training, target_training) #Object of type Classifier training model
    preds = Classifier.predict(data_testing) #Object of type Classifier predicting classes
    if(accur==True):
        print((ModelName+" accuracy: %0.4f")%(metrics.accuracy_score(target_testing, preds, normalize=True)))
    if(grph==True):
        ClassReport_Graph(Classif=Classifier, Data_train=data_training, Target_train=target_training, Data_test=data_testing,
                          Target_test=target_testing, Class=setClass, ModelName='Classifier', Accur=False, Predict=None)
    print("-"*100)
    return preds    
#    except:
#        print("CLASSIFIER-FIT/TEST_ERROR\n")
    
def Fit_Classification_Model(Classifier, Examples, Labels, Predict=False, Testing=None):
    """ Function created to receive a classification model from Scikit-Learn library and perform the fitting (and predicting) steps to be
        used in the Ensemble Stacking Meta Estimator built for parallelization.
    INPUT: classification model, training examples' features, training examples' outputs, flag for returning prediciton (default False) and
            testing examples' features (default None) """
#    try:
    Classifier.fit(Examples, Labels)
    if(Predict==True):
        Predictions = Classifier.predict(Testing)
        return Predictions
    else:
        return Classifier
#    except:
#        print("FIT-PREDICT-FUNCTION_ERROR")

def Prediction_Classification_Model(Classifier, Testing=None):
    """ Function created to receive a classification model from Scikit-Learn library and perform the prediction step to be
        used in the Ensemble Stacking Meta Estimator built for parallelization.
    INPUT: classification model and testing examples' features (default None) """
#    try:
    Predictions = Classifier.predict(Testing)
    return Predictions
#    except:
#        print("PREDICT-FUNCTION_ERROR\n")

##################################################################################################################################
'''                                                  END OF FUNCTIONS                                                   '''
##################################################################################################################################

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
##################################################################################################################################