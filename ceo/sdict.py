class ILoc(object):
    def __init__(self, d):
        self.dict = d
        self.keys = self.dict.keys()
    
    def __getitem__(self, indexs):
        r = dict()
        for key in self.dict.keys():
            r[key] = []
            for index in indexs:
                r[key].append(self.dict[key][index])
      
        return r

class Sdict(object):
    def __init__(self, it):
        self._dict = dict(it)
        self._keys = self._dict.keys()
        self.iloc = ILoc(self._dict)

    def keys(self):
        return self._keys

    def __len__(self):
        return len(self._dict[self._keys[0]])

    def __getitem__(self, key):
        return self._dict[key]
