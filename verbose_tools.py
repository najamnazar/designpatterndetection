#!/usr/bin/env python2
from collections import defaultdict
import os
import sys
import numpy as np
# Prevent .pyc file from being created
sys.dont_write_bytecode = True

def get_classes(verbose_path):
    """
    Extracts the classes from a verbose file (design pattern summary of a java script)
  
    Parameters: \n
    - verbose_path (str): file path of the verbose file 
  
    Returns: \n
    - class_ngrams (dict): Dictionary containing the names of classes (keys) and their respective ngrams (values)
    """
    # Quick catch if the file can't be opened
    try:
        verbose = open(verbose_path, "r")
    except Exception as e:
        print e
        print "\t- File could not be opened\n"

    # Initialise a dictionary (of lists) to hold the class ngrams
    class_ngrams = defaultdict(list)

    # Iterate through the file to extract class names and corresponding ngrams
    for line in verbose:
        current_class = ""      # Keep a record of the class name for the current line
        # Iterate through the items in each line
        for item in line.split():
            # Record the name of the current class
            items = item.split(":")
            if items[0] == "CLASSNAME":
                current_class = items[1]
            elif items[0] == "CLASSMETHODNGRAM":
                class_ngram = items[1]
                # Add the class_ngram to the classes list of ngrams (if not present already)
                class_ngrams[current_class].append(class_ngram)
            elif items[0] == "CLASSMETHODRETURN":
                class_ngram = items[1]
                class_ngrams[current_class].append(class_ngram)
            elif items[0] == "CLASSIMPLEMENTNAME":
                class_ngram = items[1]
                class_ngrams[current_class].append(class_ngram)

    return class_ngrams

import re
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def convert_camel_to_snake(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

def get_classes_properties(verbose_path):
    """
    Extracts the classes from a verbose file (design pattern summary of a java script)

    Parameters: \n
    - verbose_path (str): file path of the verbose file

    Returns: \n
    - class_props (dict): Dictionary containing the various class properties from the verbose files
    """
    # Quick catch if the file can't be opened
    try:
        verbose = open(verbose_path, "r")
    except Exception as e:
        print
        e
        print
        "\t- File could not be opened\n"

    # Initialise a dictionary (of lists) to hold the class properties
    class_props = defaultdict(dict)

    # Iterate through the file to extract class names and corresponding ngrams
    for line in verbose:
        current_class = ""  # Keep a record of the class name for the current line
        # Iterate through the items in each line
        for item in line.split():
            # Record the name of the current class
            items = item.split(":")
            if items[0] == "CLASSNAME":
                current_class = items[1]

            elif items[0] == "CLASSMETHODRETURN":
                method_return = items[1]
                # Add the class_ngram to the classes list of ngrams (if not present already)
                if "method_return" not in class_props[current_class]:
                    class_props[current_class]["method_return"] = list()
                class_props[current_class]["method_return"].append(method_return.lower())
                if "method_count" not in class_props[current_class]:
                    class_props[current_class]["method_count"] = 0
                class_props[current_class]["method_count"] += 1

            elif items[0] == "CLASSMETHODPARAMCOUNT":
                param_count = int(items[1])
                # Add the class_ngram to the classes list of ngrams (if not present already)
                if "param_count" not in class_props[current_class]:
                    class_props[current_class]["param_count"] = list()
                class_props[current_class]["param_count"].append(param_count)
            elif items[0] == "CLASSMETHODVARCOUNT":
                var_count = int(items[1])
                # Add the class_ngram to the classes list of ngrams (if not present already)
                if "var_count" not in class_props[current_class]:
                    class_props[current_class]["var_count"] = list()
                class_props[current_class]["var_count"].append(var_count)
            elif items[0] == "CLASSMETHODLINECOUNT":
                line_count = int(items[1])
                # Add the class_ngram to the classes list of ngrams (if not present already)
                if "line_count" not in class_props[current_class]:
                    class_props[current_class]["line_count"] = list()
                class_props[current_class]["line_count"].append(line_count)
            elif items[0] == "CLASSIMPLEMENTS":
                class_props[current_class]["implements"] = items[1]=="True"
            elif items[0] == "CLASSIMPLEMENTNAME":
                class_props[current_class]["implements_name"] = items[1]



    for k,v in class_props.iteritems():
        v["average_param_count"] = np.mean(v["param_count"])
        v["total_param_count"] = np.sum(v["param_count"])

        v["total_var_count"] = np.sum(v["var_count"])
        v["average_var_count"] = np.mean(v["var_count"])

        v["total_line_count"] = np.sum(v["line_count"])
        v["average_line_count"] = np.mean(v["line_count"])

        snaked_class_name = convert_camel_to_snake(k)
        v["class_last_name"] = snaked_class_name.split("_")[-1]
        v["class_name_words"] = snaked_class_name.split("_")

        if v["class_last_name"]!=k.lower():
            v["class_last_name_is_different"] = True
        else:
            v["class_last_name_is_different"] = False

        if "implements_name" in v:
            v["class_implements_last_name"] = convert_camel_to_snake(v["implements_name"]).split("_")[-1]
        else:
            v["class_implements_last_name"] = None

    for k, v in class_props.iteritems():
        v_dash = dict()
        for k2,v2 in v.iteritems():
            if type(v2) in [str,int,bool,float,np.float64,np.float32,np.int64,np.int32] or v2 is None or k2 in ["class_name_words","method_return"]:
                v_dash[k2] = v2
        class_props[k] = v_dash

    return class_props

def find_lbld_projects():
    """
    Constructs a list of projects that have been labelled in the output_refined.csv file
  
    Parameters: \n
    - None: input assumed to be output-refined.csv 
  
    Returns: \n
    - project_list: List of the projects containing files that have been labelled
    """
    project_list = []
    with open("output-refined.csv", "r") as lbld_data:
        for line in lbld_data:
            project, class_name, pattern, url = line.split(",")
            if project == "Project" and class_name == "Class":
                continue
            elif project not in project_list:
                project_list.append(project)

    return sorted(project_list)