"""
h5dict - HDF5-based persistent dict
===================================
"""

import numpy as np
import cPickle
import tempfile, os
import collections

import h5py

class h5dict(collections.MutableMapping):
    def __init__(self, path=None):
        if path is None:
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            tmppath = tmpfile.name
            tmpfile.close()
            self.path = tmppath
        else:
            self.path = path
        self._h5file = h5py.File(self.path, 'w')
        self._types = {}
        self._dtypes = {}

    def __contains__(self, key):
        return self._h5file.__contains__(key)

    def __iter__(self):
        return self._h5file.__iter__()

    def __len__(self):
        return len(a)

    def keys(self):
        return self._h5file.keys()

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError('\'%s\' is not in the keys' % key)

        value = self._h5file[key].value
        # If it is a single string, then it is a pickled object.
        if isinstance(value, str):
            value = cPickle.loads(value)

        return value

    def __delitem__(self, key):
        self._types.__delitem__(key)
        self._dtypes.__delitem__(key)
        self._h5file.__delitem__(key)

    def __setitem__(self, key, value):
        if key in self.keys():
            self.__delitem__(key)

        if isinstance(value, np.ndarray):
            self._h5file[key] = value
            self._types[key] = type(value)
            self._dtypes[key] = value.dtype
        else:
            self._h5file[key] = cPickle.dumps(value, protocol = -1)
            self._types[key] = type(value)
            self._dtypes[key] = None

    def value_type(self, key):
        return self._types[key]

    def value_dtype(self, key):
        return self._dtypes[key]

    def __del__(self):
        self._h5file.close()

