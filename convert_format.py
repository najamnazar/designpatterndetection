import os
from collections import defaultdict
import logging
import sys
import random

logger = logging.getLogger()

def convert_format(inDir, data):
    vocab = {}
    feat_count = 1                  ##
    freq = defaultdict(int)

    # Iterate over the methods extracted from the corpus
    for _, method_info in data:
        # What's happening here?
        # Should this be measuring the frequency of classes?
        for ff in method_info:#.split():
            # print ff.split(':')[0]
            if not ff in vocab:
                vocab[ff] = feat_count
                feat_count += 1
            freq[ff] += 1
        # exit()
    vocab["_OOV_"] = feat_count
    feat_count += 1

    oov_feat = 0
    #print "chosen_train_files", chosen_train_files
    
    vectors = []
    labels = []
    # optr = open(os.path.join(inDir, "output.txt"), "w")  # not required at this stage
    for c, f in data:
        feats = []

        for ff in f:
            if ff in vocab:
                if freq[ff] < 2:
                    feats.append(vocab["_OOV_"])
                else:
                    feats.append(vocab[ff])
            else:
                  feats.append(vocab["_OOV_"])
            if freq[ff] < 2:
                oov_feat += 1
        #print 'Ending loop'
        print_line = '-1'
        if c == 1:
            print_line = '+1'
        # svm_line = print_line                   ## Not sure what this is for
        for ff in sorted(feats):
            print_line += ' %d:1' % ff 
            # svm_line += str(ff)                 ## Not sure what this is for

        vectors.append(list(set(feats)))
        labels.append(int(print_line[:2]))
        # optr.write(print_line +'\n')
    # optr.close()
    logger.info('oov feat: ' + str(oov_feat))
    logger.info('total feat: ' + str(feat_count))

    # Randomized labels until we won't find way to label items
    #labels = [random.randint(0, 1) for i in range(len(labels))]
    for i in range(100):
        labels[i] = 0
    return labels, vectors
