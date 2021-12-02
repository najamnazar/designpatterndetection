#!/usr/bin/python
'''
This program is mainly for identifying which projects 
can be used as input, output pair. 
B
Idea is to find projects with "main" and "test" folders
where main folder consists of inputs and test folder consists
of outputs
We take test folder as output since it summarizes which methods are 
important
Usage: python summarizer.py \
          --input <input_dir_with_javacode> 
           --output <output_dir_with_input_output_pairs_identified>
           --tasks all create format extractfeat
'''

import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
import logging
import plyj.parser
import os
import sys
#from convert_format import *
from extract_features_callgraph import *
# from learning import run_train        # Not used at the moment, can be addressed later
import random

sys.setrecursionlimit(10000) # using a recursion limit to avoid crashing if the huge dataset
def check_patterns(options):
    '''
    Function to compare patterns with
    directory sturcture
    '''
    for subdir in os.listdir(options.input):
        # print "options.input", options.input

        output_files = defaultdict(list)
        for root, _, files in \
                os.walk(
                    os.path.join(options.input, subdir)):
            # check if input and output exist
            for pattern_type in options.patterns.keys():
                for pp in options.patterns[pattern_type]:
                    if "/%s/" % pp in root:
                        for fname in files:
                            # yield pattern_type, subdir, root, fname
                            output_files[pattern_type] \
                                .append(
                                (pattern_type,
                                 subdir, root,
                                 fname))
        if len(output_files["input"]) > 0:
            #   and len(output_files["output"]) > 0:        # commented since a test set is not being used atm
            for kk in output_files.keys():
                for vv in output_files[kk]:
                    yield vv


def process_dirs(options):
    '''
    Function to create input output paris
    Takes input and patterns to identify 
    input output pairs and creates output 
    folder
    We take the children dir of input dir as 
    the package name
    '''
    # check_input_pattern returns type=input/output
    # and package name and filename to copy
    for pattern_type, pname, root, fname in \
            check_patterns(options):
        package_outdir = os.path.join(options.output,
                                      pname)
        check_dir(package_outdir)
        # Code below makes a copy of the input 
        # check_dir(os.path.join(package_outdir, pattern_type))
        # invoke_command(['cp', 
        #     os.path.join(root, fname), 
        #       os.path.join(package_outdir, pattern_type)])


if __name__ == '__main__':
    # Create a argparser
    parser = argparse.ArgumentParser(
        argument_default=False,
        description="Script to \
                identify input, output pairs",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('--input', '-i',
                        action='store',
                        required=True,
                        metavar='input',
                        help='input java files dir')
    parser.add_argument('--output', '-o',
                        action='store',
                        required=True,
                        metavar='output',
                        help='output folder with input \
                     output pair')
    parser.add_argument('--tasks', '-t',
                        choices=['all', 'store', 'extractfeat', 'test', 'format'],
                        nargs='?',  # Accepts a single argument if present
                        default='all',  # If no argument is given, the default value of 'all' is used
                        help='tasks to run')
    parser.add_argument('--log', '-l',
                        action='store',
                        metavar='loglevel',
                        default='info',  # All
                        choices=['notset', 'debug', 'info',
                                 'warning'],
                        help='debug level')

    options = parser.parse_args()
    random.seed(100)
    # Set logging level
    numeric_level = getattr(logging,
                            options.log.upper())
    if not isinstance(numeric_level, int):
        ValueError('Invalid log level %s'
                   % options.log)

    # Set up logger
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    streamHandler.setLevel(numeric_level)
    logging.basicConfig(level=numeric_level, filename='detector.log', filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.addHandler(streamHandler)


    # verify input and output
    if not os.path.isdir(options.input):
        IOError('Invalid input folder %s'
                % options.input)
    logger.info('Input: %s' % options.input)

    check_dir(options.output)  # Checks if output folder exists: if not, creates one

    # All output should go to output (commented out code)
    # check_dir(
    #     os.path.join(options.output, "data"))  # Checks if 'data' folder exists in output folder: if not, creates one
    # check_dir(os.path.join(options.output,
    #                        "features"))  # Additional line: Checks if 'data' folder exists in output folder: if not, creates one
    # feat_output = os.path.join(options.output, "features")
    # options.output = os.path.join(options.output, "data")
    feat_output = options.output  # avoid refactoring for now

    logger.info('Output: %s' % options.output)

    # Configure input and output subdirectory patterns
    # these patterns would be used to find input,
    # and output code pairs
    options.patterns = {
        'input': ['main'],
        #        'output' : ['test']           # Not currently used, commented out for simplicity
    }

    # process inputs and outputs
    runAll = False
    if len(options.tasks) == 0 \
            or 'all' in options.tasks:
        runAll = True

    if runAll \
            or "store" in options.tasks:
        # find the prospective input, output pairs
        # print("Options {}".format(options))
        # process_dirs(options)  # commented to remove redundant folders
        pass

        # extract features from the input files
    if runAll \
            or "extractfeat" in options.tasks:
        check_dir(feat_output)
        parser = plyj.parser.Parser()
        # Extracts information about the classes in the provided corpus and...
        # ... whether they are invoked by other methods (using 0/1 classification)
        data = extract_features(options.input,
                                feat_output, parser)  # Changed to options.input

   #format the features into liblinear format
   # Commented by Najam
   #if runAll or \
   #      "format" in options.tasks:
   # labels, vectors = convert_format(feat_output, data)
       #print labels, vectors
       #print "len", len(labels), len([x for x in labels if x == -1])
      #  run_train(vectors, labels)                     # Not being used 
       #print "Format converted\n"
