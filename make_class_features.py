#!/usr/bin/env python2

import sys
import os
from os.path import join, getsize, splitext
import re
from gensim.models import Word2Vec, KeyedVectors
from verbose_tools import get_classes
from verbose_tools import get_classes_properties
import logging as log
from tqdm import tqdm

NDIM = 100                 ## currently set to 100 may change it to other value.
verbose_root = sys.argv[1]
log.basicConfig(level=log.DEBUG, filename='make_class_features.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = log.getLogger(__name__)

logger.info("parsing projects")
sentences = []
patterns = dict()
#data_file = "output-refined.csv"
#data_file = "input-1300.csv"
data_file = "p-mart-output-final.csv"

with open(data_file) as f:
    for line in f:
        # project,class_name,pattern,url = line.strip().split(",")
        project, class_name, pattern = line.strip().split(",")
        patterns[(project, class_name)] = pattern

file_data_store = dict()

# Iterate through all files in the input folder (verbose_root)
# for root, dirs, files in tqdm(os.walk(verbose_root)):
for root, dirs, files in os.walk(verbose_root):
    # If the are no files in the verbose root folder
    if files is None:
        logger.error("No files found in input directory")
        exit()
    for f in files:
        if ".verbose" in f:
            file_data_store[f]=dict()
            proj_name = splitext(f)[0]
            file_data_store[f]["project_name"] = proj_name
            # Groups the ngrams by class
            class_dict = get_classes(os.path.join(root, f))
            class_properties = get_classes_properties(os.path.join(root, f))

            if class_dict is None:
                logger.warning("No classes or ngrams extracted from project file {0}".format(f))
                continue
            file_data_store[f]["class_dict"] = class_dict
            file_data_store[f]["class_properties"] = class_properties

            # Appends the list of ngrams (in lower case) to a list of sentences
            for class_name, ngrams in class_dict.iteritems():
                ngram_for_class = [ngram.lower() for ngram in ngrams]
                sentences.append(ngram_for_class)

logger.info("Building Word2Vec model")
## May be able to revise these paramters for better performance
ngram_model = Word2Vec(sentences, size=NDIM, window=20, min_count=2, workers=4)

saved_items_list = set()
saved_items_dicts = list()

for f,verbose_data in file_data_store.iteritems():
    proj_name = verbose_data["project_name"]  # Name of the current project
    # Retrieve a dictionary constaining class names and corresponding ngrams
    class_dict = verbose_data["class_dict"]
    class_properties = verbose_data["class_properties"]
    # Iterate over class names and ngrams from the verbose file
    for class_name, ngrams in class_dict.iteritems():

        # The if-block below makes sure that we only keep the labelled datasets in java_class_features.txt
        # This reduces size of java_class_features.txt, before this java_class_features.txt was almost 100 MB

        if (proj_name,class_name) not in patterns:
            continue
        vector_ngram = [0.0 for i in range(NDIM)]
        ngram_count = 0
        for ngram in ngrams:
            try:
                # TODO: Check if this line works as expected
                vector_ngram += ngram_model.wv[ngram.lower()]
                ngram_count += 1
            except Exception as e:
                # log.warning("Loading Word2Vec: {0}".format(e))
                pass


        # if any ngrams were present in the trained Word2Vec embedding model
        if ngram_count > 0:
            # Normalise the vector
            vector_ngram /= float(ngram_count)

        saved_items_list.add((proj_name,class_name))
        feature_dict = dict(project_name=proj_name,class_name=class_name,)
        class_properties[class_name].pop('method_return', None)
        class_properties[class_name].pop('class_name_words', None)
        feature_dict.update(class_properties[class_name])
        feature_dict.update({"w2v_"+str(i):x for i,x in enumerate(vector_ngram)})
        saved_items_dicts.append(feature_dict)



# Printing Total number of examples identified from output-refined.csv
print "Examples identified from output-refined.csv = "+str(len(saved_items_list))

# Determining which examples are in output-refined.csv (Labelled) but missed in `.verbose` files.
patterns_keys = set(patterns.keys())
print "-"*80
print "Examples in Output-refined.csv but not in `.verbose` files = " + str(len(patterns_keys-saved_items_list))
print "-"*80
print "\nMissing Project,Class,Pattern"
for i,(proj,class_name) in enumerate(sorted(patterns_keys-saved_items_list)):
    print i,",",proj,",",class_name,",",patterns[(proj,class_name)]


print "-"*80
print "Missing Projects = "
print "-"*80

missing_projects = set([proj for proj,class_name in sorted(patterns_keys-saved_items_list)])
for proj in missing_projects:
    print proj


import pandas as pd
df = pd.DataFrame.from_records(saved_items_dicts)

#data_file = "output-refined.csv"
#data_file = "input-1300.csv"
data_file = "p-mart-output-final.csv"
# patterns = pd.read_csv(data_file,header=None,names=["project_name","class_name","pattern","url"])
patterns = pd.read_csv(data_file,header=None,names=["project_name","class_name","pattern"])
print(patterns.shape)
# pattern_repeats = patterns.groupby(["project_name","class_name"])["pattern"].count()
#patterns.drop_duplicates(["project_name","class_name"],inplace=True)
print(patterns.shape)



dataset = df.merge(patterns,on=["project_name","class_name"],how="inner")
print(dataset.shape)
# dataset_repeats = dataset.groupby(["project_name","class_name"])["pattern"].count()
# dataset_repeats = dataset_repeats[dataset_repeats>1]
#dataset.drop_duplicates(["project_name","class_name"],inplace=True)
#print(dataset.shape)
#dataset.to_csv("dataset.csv",index=False)
dataset.to_csv("P-MARt-dataset.csv",index=False)