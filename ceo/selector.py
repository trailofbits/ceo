import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        if (type(X) == np.ndarray):
            return X
        return X.toarray()

    def fit_transform(self, X, y=None, **fit_params):
        #self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    #def get_params(self, deep=True):
    #    return []

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        #print "key:", key
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X, y=None):
        #print data_dict
        return X[self.key]


def make_features_pipeline(features, vectorizers, clf):
    transformer_list = []
    for feature in features:
        vectorizer = vectorizers[feature]
        transformer = (feature, Pipeline([
                ('selector', ItemSelector(key=feature)),
                (feature+"-vectorizer", vectorizer),
                ('todense', DenseTransformer()),
            ]))
        transformer_list.append(transformer)

    return  Pipeline([
                ('preprocesor', FeatureUnion(transformer_list=transformer_list)),
                ('classifier', clf),
                ])

    #pipeline = FeatureUnion(transformer_list=transformer_list)
    #return pipeline
   
