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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer as skCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as skTfidfVectorizer
from sklearn.externals import joblib

import networkx as nx
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

# TODO: find a better place for this
csv.field_size_limit(sys.maxsize) 

from ceo.tools import manticore_policies, manticore_dist_symb_bytes
from ceo.tools import grr_mutators 
from ceo.features import features_list

class Vectorizer:
    '''
        Abstract class for vectorization of features
    ''' 


def _tokenizer(s):
    return filter(lambda x: x != '' and len(x) <= 32, s.split(" "))


class AFLParamVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary_ = dict()
        self.vocabulary_["has_params"] = 0 

    def fit(self, x, y=None):
        pass

    def fit_transform(self, x, y=None):
        return self.transform(x)

    def transform(self, x, y=None):
        arr = [[0]]*len(x)
        return np.array(arr)

class GrrParamVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mutators = grr_mutators
        self.vocabulary_ = dict()
        for i,mutator in enumerate(self.mutators):
            self.vocabulary_[mutator] = i

    def fit(self, x, y=None):
        pass

    def fit_transform(self, x, y=None):
        return self.transform(x)

    def transform(self, xs, y=None):
        arr = []
        for x in xs:
            ret = []
        
            strings = self.mutators
            v = [0]*len(strings)
            v[strings.index(x["mutator"])] = 1
            ret.extend(v)
            arr.append(ret)

        return np.array(arr)


class MCoreParamVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary_ = dict()
        i = 0

        for x in manticore_policies:
          self.vocabulary_["policy="+x] = i
          i = i + 1

        for x in manticore_dist_symb_bytes:
          self.vocabulary_["dist_symb_bytes="+x] = i
          i = i + 1

        self.vocabulary_["rand_symb_bytes"] = i

    def fit(self, x, y=None):
        pass

    def fit_transform(self, x, y=None):
        return self.transform(x, y)

    def transform(self, xs, y=None):
        arr = []
 
        for x in xs:
            ret = []
            #print "x",x   
            # TODO: use constants from tool.py
            strings = ["random", "uncovered", "branchlimited"] 
            v = [0]*len(strings)
            v[strings.index(x["policy"])] = 1
            ret.extend(v)

            strings = ["sparse", "dense"]
            v = [0]*len(strings)
            v[strings.index(x["dist_symb_bytes"])] = 1
            ret.extend(v)
 
            ret.append(x["rand_symb_bytes"])
            arr.append(ret)
 
        return np.array(arr)


class GraphVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary_ = dict() 

    def fit(self, x, y=None):
        pass

    def fit_transform(self, x, y=None):
        return self.transform(x, y)

    def transform(self, gs, y=None):
        m = []
        for raw_edges in gs:
            edges = []
            for x in raw_edges.split(" "):
                edges.append(x.split(","))

            H = nx.DiGraph()

            for (x,y) in edges:
                H.add_edge(x,y)
        
            nnodes = H.number_of_nodes()

            #pos=nx.nx_agraph.graphviz_layout(H)
            #nx.draw(H)
            #plt.show()

            #print H.number_of_nodes()
            #print nx.info(H)
            """
            pathlengths=[]
            for v in H.nodes():
                spl=nx.single_source_shortest_path_length(H,v)
                for p in spl.values():
                    pathlengths.append(p)

            dist={}
            for p in pathlengths:
                if p in dist:
                    dist[p]+= 1
                else:
                    dist[p]= 1

            u = [0]*1024
            if nnodes > 0:
                for i in range(1024):
                    if i in dist:
                        u[i] = dist[i] / float(nnodes)
            """
            #print u 
            v = nx.degree_histogram(H)
            v = v[:18]
            v = v + [0]*(18-len(v))
            #del v[2]
            #del v[0]
            total = sum(v)
            if total > 0:
                for i,x in enumerate(v):
                    v[i] = x / float(total)  
            m.append(v)
            #print v
        return np.array(m)


from fastText import load_model
ftmodel = load_model('/home/gustavo/data/wiki.en.bin')
#ftwords = ftmodel.get_words()


class SentVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary_ = dict() 

    def fit(self, x, y=None):
        pass

    def fit_transform(self, x, y=None):
        return self.transform(x, y)

    def transform(self, texts, y=None):
        ret = []
        for text in texts:

            # parsing of some of the words we can find
            words = set(map(str.lower, _tokenizer(text)))
            for word in set(_tokenizer(text)):
                for i,letter in enumerate(word):
                    if i == len(word)-1:
                        break
                    if letter.islower() and word[i+1].isupper():
                        words.add(str.lower(word[:i+1]))
                        words.add(str.lower(word[i+1:]))

            available_words = ftmodel.get_words()
            acc = np.zeros(300)
            fwords = set()
            for word in words:
                if word in available_words:
                    fwords.add(word)

            for word in fwords:
                v = np.array(ftmodel.get_word_vector(word))
                acc = acc + v

            if len(fwords) > 0:
                acc = acc / len(fwords)

            ret.append(acc)
 
        return np.array(ret)

"""

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
"""

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

def init_vectorizers(selected_features):
    use_idf = False
    normalizer = None
    vectorizers = dict()
    syscalls = ["_receive","_transmit", "_allocate", "_deallocate", "_fdwait","_terminate","_random"]

    if "insns" in selected_features:
        vectorizers["insns"] = skTfidfVectorizer(ngram_range=(3,3), max_features=500,
                                                    tokenizer=_tokenizer, lowercase=False, norm=normalizer, use_idf=use_idf,
                                                    decode_error="replace")

    if "syscalls" in selected_features:
        vectorizers["syscalls"] = skTfidfVectorizer(ngram_range=(3,3), max_features=500, 
                                          tokenizer=_tokenizer, lowercase=False, norm=normalizer, use_idf=use_idf,
                                          #vocabulary=syscalls, 
                                          decode_error="replace")

    #exec_vectorizers["reads"] = SeriesVectorizer() 
    #exec_vectorizers["writes"] = SeriesVectorizer() 
    #exec_vectorizers["allocs"] = SeriesVectorizer() 
    #exec_vectorizers["deallocs"] = SeriesVectorizer()
 
    if "transmited" in selected_features: 
        vectorizers["transmited"] = SentVectorizer()
                                     #skTfidfVectorizer(ngram_range=(1,1), max_features=500,
                                     #               tokenizer=_tokenizer, lowercase=True, norm=normalizer, use_idf=use_idf,
                                     #               min_df=2, decode_error="replace")

    if "visited" in selected_features: 
        vectorizers["visited"] = GraphVectorizer()

    vectorizers["afl"]  = AFLParamVectorizer()
    vectorizers["mcore"] = MCoreParamVectorizer()
    vectorizers["grr"] = GrrParamVectorizer()

    return vectorizers #exec_vectorizers, param_vectorizers


"""
def vectorize(x_train, option, features, verbose=0):
  
    data = []
    exec_vectorizers, param_vectorizers = init_vectorizers(x_train)
    if verbose > 0:
        print exec_vectorizers
        print param_vectorizers

    exec_features = x_train["exec_features"] 
    param_features =  x_train["param_features"]

    for i in range(len(param_features)):
       
        v = []

        for feature in features:
            x = exec_features[feature][i]
            vectorizer = exec_vectorizers[feature]
            y = vectorizer.transform([x])
            #print y.shape, 
            v.append(y)
        
       
        x = param_features[i]
        if len(x) > 0:
            vectorizer = param_vectorizers[option]        
            y = vectorizer.transform([x])
            #print y.shape
            v.append(y)
        data.append(np.concatenate(v, axis=1).flatten())

    return data 
"""
