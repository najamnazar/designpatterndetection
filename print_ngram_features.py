#!/usr/bin/env python

import sys
from gensim.models import Word2Vec, KeyedVectors
import logging as log
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

model_java_file = sys.argv[1]
model_glove_file = sys.argv[2]
output_java_file = sys.argv[3]
output_glove_file = sys.argv[4]

log.info("reading model from file {0}".format(model_java_file))
model_java = Word2Vec.load(model_java_file)
log.info("reading model from file {0}".format(output_glove_file))
model_glove = KeyedVectors.load(model_glove_file)

output_java = open(output_java_file, "w")
output_glove = open(output_glove_file, "w")

log.info("writing features to file {0}".format(output_glove_file))
for word in tqdm(model_glove.index2word):
    output_glove.write("{0} {1}\n".format(word.encode('utf-8'), " ".join(["{0:.6f}".format(x) for x in model_glove[word]])))

log.info("writing features to file {0}".format(output_java_file))
for word in tqdm(model_java.wv.index2word):
    output_java.write("{0} {1}\n".format(word, " ".join(["{0:.6f}".format(x) for x in model_java[word]])))
