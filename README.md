# Design-Pattern-detection

This application identifies design pattern instances from a given dataset. Currently, the code is compatible with Python 2.7

#### Note: The files below require the matplotlib package for plotting and readable output

## Data Source
- [Java file Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/)


## Dataset Generation
The dataset used for the classification is generated in a series of steps. First, make_ngram.py is used to construct a Word2Vec model using a set of files (verbose files) containing class names and ngrams associated with the class. make_glove.py performs a similar task, but instead constructs a GloVe model given the verbose files. The word embedding model is used to construct a dataset by running make_class_features.py followed by 

### detector.py

Outputs `.verbose` files in the output directory taking `.java` files from input directory

Usage:
```python
python detector.py --input ./input --output ./output --tasks all
``` 

### make_class_features.py

Outputs vector representation Java classes from the ngram model in `dataset.csv` along with labels merged from `output-refined.csv`

Usage:  
```python
python make_class_features.py [VERBOSE_ROOT]
```

Arguments:  
- VERBOSE_ROOT: root of the directory containing the verbose files. 

NOTE: This script uses the Word2Vec implementation in the [Gensim package](https://github.com/RaRe-Technologies/gensim "Gensim Github Repo"). 
And may need to install sudo apt-get install -y liblzma-dev to run it for linux machines.


## Classification
Classification is performed in the python files listed below. The file calls the dataset.csv file, which is used to train a machine learning model and generate predictions. 

Each script outputs 
- confusion matrix csv and plot
- Per class Precision, Recall barplots
- report containing the precision, recall, and f-score values for the prediction.


### classifier.py
Uses a various to classify the design patterns contained in the test files.  

Usage:
```python
python classifier.py [CLASSIFIERNAME]
```  

NOTE: Assumes the dataset is called 'dataset.csv'  

Arguments: 

- CLASSIFIERNAME - Which Classification algorithm you want to use.

CLASSIFIERNAME supported are
- RF: RandomForest
- SVM: Support Vector Machine
- ADABOOST: Adaboost
- ADABOOST_LOGISTIC: Adaboost
- LOGISTIC: Logistic Regression
- GBTREE: GradientBoosted Tree
- RIDGE: Ridge classifier
- VOTER: Voting classifier composed of RF, SVM, ADABOOST


## Miscallenous Scripts [currently not used]

### print_ngram_features.py

Prints the vector representations of the ngrams in readable format.

Usage:  
print_ngram_features.py [MODEL_JAVA_FILE] [MODEL_GLOVE_FILE] [OUTPUT_JAVA_FILE] [OUTPUT_GLOVE_FILE]

Arguments:  
<li>MODEL_JAVA_FILE: file name of the ngram vector model  
<li>MODEL_GLOVE_FILE: file name of the GloVe vector model  
<li>OUTPUT_JAVA_FILE: text file to write the vector representations of the ngrams  
<li>OUTPUT_GLOVE_FILE: text file to write the vector representations of GloVe


