'''
Script to extract features and create
liblinear format files 
creates two files 
   1) feat.verbose
       describes all the features and classes
   2) feat.liblinear
       liblinear format
'''
import os
import plyj.model as m
import logging
import subprocess
import sys, traceback
from re import finditer
from call_graph import *

logger = logging.getLogger()

# Moved this function from summarizer to stop some circular references
def check_dir(adir):
    #creates directories
    if not os.path.isdir(adir):
        invoke_command(['mkdir', adir])

# Moved this function from summarizer to stop some circular references
def invoke_command(cmd_array):
   logging.debug('Run command: %s' 
        %(' '.join(cmd_array)))
   subprocess.check_call(cmd_array)


def _f(kk, vv, checkCamelCase=False):
    if not vv:
        return "NONEFEATURE"
    if checkCamelCase:
        feats = []
        for cc in camel_case_split(vv):
            feats.append("%s:%s" %(kk, cc))
        return " ".join(feats)
    else:
        return "%s:%s" %(kk, vv)

def extract_features(inDir, outDir, parser):
    '''
    inDir has projects/(input, output) pairs
    We are going to do the following 
      for every project
        for every file in input and output
            1) use the ply parser to parse the file
            2) extract classname, method features
            3) use the output to select the class values
    '''
    features_to_extract = [
       "class", #class name
       "class_modifiers", #class modifiers public, private
       "class_implements", #interfaces
       "class_extends", #extends some class
       "method_name_ngram", #ngram word feature
       "method_type_param", #type param
       "method_param", #inputs
       "method_return_type", #return type
       "method_num_variables", #number of variables
       "method_num_function", #number of func calls
       "method_num_lines", #size of method
       "method_incoming_functions", #incoming function size and names
       "method_incoming_function_names", #incoming function size and names
       "method_outgoing_functions", #outgoing function size and names
       "method_outgoing_function_names", #outgoing function size and names
       "method_body_line_types" #statement types
    ]
    training_size = 0               ## May be able to get rid of this if unused
    proj_counter = 0                # Counter for the number of projects
    file_counter = 0                # Counter for the number of files
    
    proj_file_list = defaultdict(list)   # Dictionary to hold the names of files and their projects 
    method_summaries = []                # List to hold the features extracted from each file
    
    # Iterate over all the projects in the corpus given by inDir
    for proj in os.listdir(inDir):
        proj_counter += 1           # Increment the project counter
        method_feature_list = []    ## List to store the features found
        
        rel_path = os.path.join(inDir, proj)    # Path of current project
        
        # Debug Output:
        # print "PROJECT NAME:\t" + proj
        # print "  PROJECT PATH:\t" + rel_path

        # Finds the root, directories and files in the current path
        for root, drs, files in os.walk( \
                  os.path.join(rel_path)):
            
            for input_file in files:
                file_counter += 1                       # Increment the file counter
                proj_file_list[proj].append(input_file) # Store the filename

                # Debugging:
                # print "root {0} drs {1} input_file {2}".format(root, drs, input_file)
                
                # Ignores non-java files
                if not input_file.endswith(".java"):
                    continue

                input_file_abs = os.path.join(root, input_file)

                # Debugging:
                # print "Input file:\t" + input_file
                # print 'Input_file_abs', input_file_abs

                try:
                    # Generate a callgraph from the current file
                    cgraph = Callgraph.from_file(input_file_abs, parser)
                    # Grab a list of methods from the callgraph
                    method_feature_list += \
                        from_tree(cgraph, features_to_extract)
                except Exception as e:
                    logger.error("ERROR: " + e.message)
                    logger.error('\t- Errored path:\t' + input_file_abs)       # Changed from python 3 to python 2 syntax for consistency
            
        #print 'Method feature list', method_feature_list
        #print "Here we go!!!???!"
        training_size += len(method_feature_list)           ##

        method_names_in_summary = []
        #print "Starting loop1"
        #print "Trying to loop over: ", os.path.join(rel_path, "output")
        for root, drs, files in os.walk( \
                  os.path.join(rel_path)):
            for output_file in files:
                output_file_abs = os.path.join(root, output_file)
                try:
                    tree = parser.parse_file(output_file_abs)
                except:
                    logger.error('errored path'+ input_file_abs)
                    print "Errored cgraph" +str(cgraph)
                    #logger.error('errored cgraph'+ cgraph)
                    continue

                method_names_in_summary += add_class_labels(tree, 
                                        method_feature_list)
        
        #print "method_names_in_summary", method_names_in_summary

        #class_labels = {}
        # print "Now opening/creating:\t" + os.path.join(outDir, "%s.verbose" % proj)

        # Write summary of the current project file to a text file 
        optr = open(os.path.join(outDir, "%s.verbose" % proj), "w")
        for mm in method_feature_list:
            # If the method has been selected for inclusion in the summary
            if mm[0] in method_names_in_summary:
                 _line = "1\t%s\n" %(" ".join(mm[1]))
                 optr.write(_line)
                 method_summaries.append((1, mm[1]))                
            else: 
                 _line = "0\t%s\n" %(" ".join(mm[1]))
                 optr.write(_line)
                 method_summaries.append((0, mm[1]))

        optr.close()

    # Write the summaries of all files in corpus to a file 
    #Currently not needed...
    #full_out = open(os.path.join(outDir, "full_corpus.verbose"), "w")
    #for item in method_summaries:
    #    full_out.write("%s\t%s\n" %(item[0], " ".join(item[1])))
    #full_out.close()

    # Write some details about the corpus to a file
    summary = open(os.path.join(outDir, "corpus_summary.csv"), "w")
    summary.write("Number of projects, Number of files\n")
    summary.write(str(proj_counter) + ',' + str(file_counter) + '\n\n')
    summary.write("PROJECT, CLASS\n")
    for project, file_list in proj_file_list.iteritems():
        for file in file_list:
            summary.write("%s, %s \n" %(project, file))
    summary.close()

    return method_summaries

def add_class_labels(tree,
       method_feature_list):
    #print "Tree type declarations is: ", tree.type_declarations
    #print "Method feature list: ",method_feature_list
    method_names_in_summary = []
    try:
        #if tree is not None and tree.type_declarations is not None:
        if tree is not None:
          # For the objects in the body of the code
          # if type_decl != NONE and type_decl != ""
          #print "Tree type declaration is: ", tree.type_declarations
          for type_decl in tree.type_declarations:
            if not hasattr(type_decl, 'body'):
                continue
            # Construct a list of methods in the body of the code
            methods = [decl for decl in type_decl.body \
                         if type(decl) is m.MethodDeclaration]
            # Iterate through the above list of methods
            for method_decl in methods:
                try:
                    method_string = str(method_decl)    # Cast to string
                    if "MethodInvocation" in method_string:
                        ## Add the name of the method invoked by the current method??
                        method_names_in_summary += \
                          [x.split("'")[0] for x in \
                            method_string.split("MethodInvocation(name='")[1:]]
                except Exception as e:
                    logger.error(str(e))# added str(e) to remove TypeError
                    logger.error("\t- Error occured in add_class_labels function")
                    continue
        # else:
    #     print "There is a stupid error"
    #     continue
    except Exception as e:
        logger.error(sys.stderr)
        # traceback.print_exc()
        # traceback.format_exc()
    return method_names_in_summary

def from_tree(cgraph, fte):
    '''
    takes a parse_tree and list of features to extract
    '''
    method_feature_list = []
    tree = cgraph.tree
    if not tree:
        return method_feature_list
    for type_decl in tree.type_declarations:
        #print 'TYPE DECL!!!!!!!!!', type_decl, '\n'
        class_feature_list = []
        if not hasattr(type_decl, 'name'):
            return method_feature_list
        class_name = type_decl.name
        class_feature_list += handle_class(fte,
                          type_decl)
        class_feature_list += handle_class_modifier(fte,
                          type_decl)
        class_feature_list += handle_class_implements(fte,
                          type_decl)
        class_feature_list += handle_class_extends(fte,
                          type_decl)
        for method_decl in cgraph.nodes:
            feature_list = []
            method_name = method_decl.name
            feature_list += handle_method_ngram(fte,
                               method_decl)
            feature_list += handle_method_return(fte,
                              method_decl)
            feature_list += handle_method_param(fte,
                              method_decl)
            feature_list += handle_method_stats(fte,
                              cgraph, method_decl)
            method_feature_list.append(
              (method_name, 
                  class_feature_list + feature_list))
    return method_feature_list 
    
def handle_method_ngram(fte, method_decl):
    feature_list = []
    method_name = method_decl.name
    if "method_name_ngram" in fte:
        feature_list.append(
          _f("CLASSMETHODNGRAM",
               method_name, checkCamelCase=True))
    return feature_list

def handle_method_return(fte, method_decl):
   feature_list = []
   if "method_return_type" in fte:
       if method_decl.return_val is not None:
            feature_list.append(
                   _f("CLASSMETHODRETURN",
                       method_decl.return_val, 
                         checkCamelCase=True))
   return feature_list

def handle_method_param(fte, method_decl):
    feature_list = []
    if "method_type_param" in fte:
        pkeys = method_decl.params.keys()
        pvalues = method_decl.params.values()
        feature_list.append(_f("CLASSMETHODPARAMCOUNT", 
            str(len(pkeys))))
        for kk, vv in zip(pkeys, pvalues):
            param_name = kk
            param_val = type2str(vv)
            if param_name is not None:
                feature_list.append(
                   _f("CLASSMETHODPARAMNAME",
                     param_name, checkCamelCase=True))
            if param_val is not None:
                feature_list.append(
                   _f("CLASSMETHODPARAMTYPE",
                     param_val, checkCamelCase=True))

    return feature_list     

def handle_method_stats(fte, cgraph, method_decl):
    feature_list = []
    var_count = len(method_decl.params.keys())
    lines_count = len(method_decl.body)
    
    if "method_num_variables" in fte:
        feature_list.append(
          _f("CLASSMETHODVARCOUNT",
             str(var_count)))
    if "method_num_lines" in fte:
        feature_list.append(
          _f("CLASSMETHODLINECOUNT",
             str(lines_count)))
    if "method_body_line_types" in fte:
        for st_type in method_decl.body:
            feature_list.append(
              _f("CLASSMETHODLINETYPE",
                 type2str(st_type)))
    if "method_incoming_functions" in fte:
        in_count = len(cgraph.graph['in'][
                       method_decl.name])
        feature_list.append(
          _f("CLASSMETHODINCOMING",
             str(in_count)))
    if "method_incoming_function_names" in fte:
        for in_name in cgraph.graph['in'][
                       method_decl.name]:
            feature_list.append(
              _f("CLASSMETHODINCOMINGNAME",
                 in_name, checkCamelCase=True))
    if "method_outgoing_functions" in fte:
        out_count = len(cgraph.graph['out'][
                       method_decl.name])
        feature_list.append(
          _f("CLASSMETHODOUTGOING",
             str(out_count)))
    if "method_outgoing_function_names" in fte:
        for out_name in cgraph.graph['out'][
                       method_decl.name]:
            feature_list.append(
              _f("CLASSMETHODOUTGOINGNAME",
                 out_name, checkCamelCase=True))
        
    return feature_list

 
def handle_class(fte, type_decl):
        feature_list = []
        class_name = type_decl.name
        if "class" in fte:
            feature_list.append(_f("CLASSNAME", 
                         class_name))
        return feature_list

def handle_class_modifier(fte, type_decl):
        feature_list = []
        if "class_modifiers" in fte:
            for mm in type_decl.modifiers:
                if type(mm) is str:
                    feature_list.append(_f("CLASSMODIFIER",
                            mm))
        return feature_list

def handle_class_implements(fte, type_decl):
        feature_list = []
        if "class_implements" in fte:
            try:
                if len(type_decl.implements) is 0:
                    feature_list.append(_f("CLASSIMPLEMENTS",
                                 "False"))
                else:
                    feature_list.append(_f("CLASSIMPLEMENTS",
                                 "True"))
                    for tt in type_decl.implements:
                        feature_list.append(_f("CLASSIMPLEMENTNAME",
                              tt.name.value, checkCamelCase=True))
            except: #its an interface
                do_nothing = 1  
        return feature_list

def handle_class_extends(fte, type_decl):
        feature_list = []
        if "class_extends" in fte:
            if not hasattr(type_decl, 'extends'):
                return feature_list
 
            if type_decl.extends is not None:
                feature_list.append(_f("CLASSEXTENDS",
                           "True"))
                if type(type_decl.extends) is list:
                    if len(type_decl.extends) > 0:
                        for tt in type_decl.extends:
                            feature_list.append(
                              _f("CLASSEXTENDNAME",
                                   tt.name.value,
                                     checkCamelCase=True))
                else:
                    feature_list.append(_f("CLASSEXTENDNAME",
                                   type_decl.extends.name.value,
                                     checkCamelCase=True))
                                  
            else:
                feature_list.append(_f("CLASSEXTENDS",
                              "False"))
        return feature_list
        
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def type2str(s):
    if type(s) == str:
        return s
    else:
       return str(type(s)).split("'")[1]