import os
import time
import sys
import re
from subprocess import call
import numpy as np
import pickle
import csv
import gzip

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer as skCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as skTfidfVectorizer
from sklearn.externals import joblib

import networkx as nx
import matplotlib as mpl
#mpl.use('Pdf')
import matplotlib.pyplot as plt

from ceo.tools import manticore_policies, manticore_dist_symb_bytes
from ceo.tools import grr_mutators 
from ceo.features import features_list

class Vectorizer:
    '''
        Abstract class for vectorization of features
    ''' 


def _tokenizer(s):
    return filter(lambda x: x != '', s.split(" "))


class AFLParamVectorizer(Vectorizer):
    def __init__(self):
        pass

    def fit_transform(self, x):
        return self._vectorizer.transform(x)

    def transform(self, x):
        return np.array(x)

class GrrParamVectorizer(Vectorizer):
    def __init__(self):
        self.mutators = grr_mutators

    def fit_transform(self, x):
        return self._vectorizer.transform(x)

    def transform(self, x):
        x = x[0]
        ret = []
        
        strings = self.mutators
        v = [0]*len(strings)
        v[strings.index(x[0])] = 1
        ret.extend(v)

        return np.array([ret])


class MCoreParamVectorizer(Vectorizer):
    def __init__(self):
        pass

    def fit_transform(self, x):
        return self._vectorizer.transform(x)

    def transform(self, x):
        x = x[0]
        #print "x",x
        ret = []
        
        strings = ["random", "uncovered", "branchlimited"] 
        v = [0]*len(strings)
        v[strings.index(x[0])] = 1
        ret.extend(v)

        strings = ["sparse", "dense"]
        v = [0]*len(strings)
        v[strings.index(x[1])] = 1
        ret.extend(v)
 
        ret.append(x[2])
        return np.array([ret])


class GraphVectorizer(Vectorizer):
    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, edgess):
        edges = edgess[0]

        H = nx.DiGraph()

        for (x,y) in edges:
            H.add_edge(x,y)

        pos=nx.nx_agraph.graphviz_layout(H)
        nx.draw(H,pos)
        plt.show()

        print H.number_of_nodes()
        print nx.info(H)
        v = nx.degree_histogram(H)
        v = v[:16]
        v = v + [0]*(16-len(v))
        print v
        return np.array([v])

class SeriesVectorizer(Vectorizer):
    def __init__(self):
        self.ranges = [(0,0)]
        self.ys = dict()
        for i in range(31):
            k = (2**i,2**(i+1))
            self.ranges.append(k) 

    def fit(self, x):
        pass

    def transform(self, xss):
        xs = xss[0]
        ys = dict()
        
        for k in self.ranges:
            ys[k] = 0

        for x in xs:
            for (r0,r1) in self.ranges:
                if x >= r0 and x <= r1:
                    ys[(r0,r1)] = ys[(r0,r1)] + 1

        v = [0] * len(self.ranges)

        for i,k in enumerate(self.ranges):
            v[i] = ys[k]

        return np.array([v])
            
        
        
 

class TFIDFVectorizer(Vectorizer):
    def __init__(self, ngram_range, max_features, vocabulary=None):
        self._vectorizer = skTfidfVectorizer(ngram_range=ngram_range, max_features=max_features, 
                                          tokenizer=_tokenizer, lowercase=True,
                                          vocabulary=vocabulary)

    def fit_transform(self, x):
        return self._vectorizer.fit_transform(x)

    def fit(self, x):
        return self._vectorizer.fit(x)

    def transform(self, x):
        return self._vectorizer.transform(x).toarray()
   
class CountVectorizer(Vectorizer):
    def __init__(self, ngram_range, max_features):
        self._vectorizer = skCountVectorizer(ngram_range=ngram_range, max_features=max_features, tokenizer=_tokenizer, lowercase=True)

    def fit_transform(self, x):
        return self._vectorizer.fit_transform(x)

    def fit(self, x):
        return self._vectorizer.fit(x)

    def transform(self, x):
        return self._vectorizer.transform(x).toarray()

class Sent2VecVectorizer(Vectorizer):
    def __init__(self, dims, ngrams):
        self._dims = dims
        self._ngrams = ngrams
        self._fasttext_exec_path = "fasttext"

    def fit_transform(self, sentences):
        """Arguments:
        - sentences: a list of preprocessed sentences
        """

        #./fasttext sent2vec -input wiki_sentences.txt -output my_model -minCount 8 -dim 700 
        #-epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 
        #-minCountLabel 20 -bucket 4000000
 
        timestamp = str(time.time())
        train_path = os.path.abspath('./'+timestamp+'_fasttext.test.txt')
        model_path = os.path.abspath('./'+timestamp+'_fasttext.model') 
        embeddings_path = os.path.abspath('./'+timestamp+'_fasttext.embeddings.txt')
        self._dump_text_to_disk(train_path, sentences)

        call(self._fasttext_exec_path +
          ' sent2vec '+' -input '+ train_path +
          ' -output ' + model_path + ' -minCount 1 ' 
          ' -dim ' + str(self._dims) + ' -epoch 5 -lr 0.2 '
          ' -wordNgrams ' + str(self._ngrams) + 
          ' -loss ns -neg 10 -thread 20 -dropoutK 2'  
          , shell=True) 

        embeddings_path.append(".bin") 

        call(self._fasttext_exec_path +
          ' print-sentence-vectors ' +
          model_path + ' < '+
          train_path + ' > ' +
          embeddings_path, shell=True)
        embeddings = self._read_embeddings(embeddings_path)
        #os.remove(test_path)
        #os.remove(embeddings_path)
        print (len(sentences), len(embeddings))
        return np.array(embeddings)

    def _read_embeddings(self, embeddings_path):
        """Arguments:
            - embeddings_path: path to the embeddings
        """
        with open(embeddings_path, 'r') as in_stream:
            embeddings = []
            for line in in_stream:
                line = map(float, line.split(" ")) #'['+line.replace(' ',',')+']'
                embeddings.append(line)
            return embeddings
        return []

    def _dump_text_to_disk(self, file_path, X, Y=None):
        """Arguments:
        - file_path: where to dump the data
        - X: list of sentences to dump
        - Y: labels, if any
        """
        with open(file_path, 'w') as out_stream:
            if Y is not None:
                for x, y in zip(X, Y):
                    out_stream.write('__label__'+str(y)+' '+x+' \n')
            else:
                for x in X:
                    out_stream.write(x+' \n')

def init_vectorizers():
    filename = "boot.csv.gz"

    exec_vectorizers = dict()
    param_vectorizers = dict()

    insns_idx = 1
    syscalls_idx = 2
    reads_idx = 3
    writes_idx = 4
    allocs_idx = 5
    deallocs_idx = 6

    #programs = []
    insns = []
    syscalls = []
    csv.field_size_limit(sys.maxsize)
    raw_features = dict()

    for name in features_list:
        raw_features[name] = []

    with gzip.open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            #programs.append(row[program_idx]) 
            raw_features["insns"].append(row[insns_idx])
            raw_features["syscalls"].append(row[syscalls_idx]) 
            raw_features["reads"].append(map(int, row[reads_idx].split(","))) 
            raw_features["writes"].append(map(int, row[writes_idx].split(","))) 
            raw_features["allocs"].append(map(int, row[allocs_idx].split(","))) 
            raw_features["deallocs"].append(map(int, row[deallocs_idx].split(","))) 

    #print raw_features
    #assert(0)
    syscalls = ["_receive","_transmit", "_allocate", "_deallocate", "_fdwait","_terminate","_random"]

    exec_vectorizers["insns"] = TFIDFVectorizer(ngram_range=(3,3), max_features=500)
    exec_vectorizers["syscalls"] = TFIDFVectorizer(ngram_range=(1,1), max_features=500, vocabulary=syscalls)
    exec_vectorizers["reads"] = SeriesVectorizer() 
    exec_vectorizers["writes"] = SeriesVectorizer() 
    exec_vectorizers["allocs"] = SeriesVectorizer() 
    exec_vectorizers["deallocs"] = SeriesVectorizer() 
    exec_vectorizers["visited"] = GraphVectorizer()

    exec_vectorizers["insns"].fit(raw_features["insns"])
    exec_vectorizers["syscalls"].fit(raw_features["syscalls"])
    exec_vectorizers["reads"].fit(raw_features["reads"])
    exec_vectorizers["writes"].fit(raw_features["writes"])
    exec_vectorizers["allocs"].fit(raw_features["allocs"])
    exec_vectorizers["deallocs"].fit(raw_features["deallocs"])

    param_vectorizers["afl"]  = AFLParamVectorizer()
    param_vectorizers["mcore"] = MCoreParamVectorizer()
    param_vectorizers["grr"] = GrrParamVectorizer()

    #print raw_features["writes"][0]
    #print vectorizer_writes.transform(raw_features["writes"][0])
    #assert(0)
    
    #exec_vectorizers.append(vectorizer_insns)
    #exec_vectorizers.append(vectorizer_syscalls)
    #exec_vectorizers.append(vectorizer_reads)
    #exec_vectorizers.append(vectorizer_writes)
    #exec_vectorizers.append(vectorizer_allocs)
    #exec_vectorizers.append(vectorizer_deallocs)

    #param_vectorizers["afl"] = vectorizer_afl_params 
    #param_vectorizers["mcore"] = vectorizer_mcore_params
    #param_vectorizers["grr"] = vectorizer_grr_params
 
    return exec_vectorizers, param_vectorizers
