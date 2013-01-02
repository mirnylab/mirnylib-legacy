#(c) 2012 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Anton Goloborodko (golobor@mit.edu),
# Maksim Imakaev (imakaev@mit.edu)


"""
h5dict - HDF5-based persistent dict
===================================
"""

import numpy as np
import cPickle
import tempfile
import os
import collections
import logging

import h5py

log = logging.getLogger(__name__)


class h5dict(collections.MutableMapping):
    self_key = '_self_key'

    def __init__(self, path=None, mode='a', autoflush=True, in_memory=False,
                 read_only=False):
        '''A persistent dictionary with data stored in an HDF5 file.

        Parameters:
        path : str
            The path to an HDF5 file. If None, than create a temporary file
            that will be deleted with the object.

        mode : str
            'r'  - Readonly, file must exist
            'r+' - Read/write, file must exist
            'w'  - Create file, truncate if exists
            'w-' - Create file, fail if exists
            'a'  - Read/write if exists, create otherwise (default)

        in_memory : bool
            if True, than the object is stored in the memory and not saved
            to the disk.
        '''
        self.read_only = (mode == 'r')

        if in_memory:
            tmpfile = tempfile.NamedTemporaryFile()
            tmppath = tmpfile.name
            tmpfile.close()
            self.path = tmppath
            self._h5file = h5py.File(tmppath, driver='core',
                                     backing_store=False)
            self.__self_load__()
            self.autoflush = False
            self.is_tmp = False  # In-memory h5dict doesn't have any tmp files.

        else:
            if path is None:
                tmpfile = tempfile.NamedTemporaryFile(delete=False)
                tmppath = tmpfile.name
                tmpfile.close()
                self.path = tmppath
                self.is_tmp = True
            else:
                self.path = os.path.abspath(os.path.expanduser(path))
                self.is_tmp = False
            if os.path.isfile(self.path):
                if not os.access(self.path, os.R_OK):
                    raise Exception('Cannot read {0}.'.format(self.path))
                if not self.read_only and not os.access(self.path, os.W_OK):
                    raise Exception('The file {0} is read-only, set mode=\'r\'.'.format(self.path))
            self._h5file = h5py.File(self.path, mode)
            self.__self_load__()
            self.autoflush = autoflush

    def __self_dump__(self):
        if self.self_key in self._h5file.keys():
            self._h5file.__delitem__(self.self_key)

        data = {'_types': self._types, '_dtypes': self._dtypes}
        dsetData = cPickle.dumps(data, protocol=0)
        self._h5file.create_dataset(name=self.self_key, data=dsetData)

    def __self_load__(self):
        if self.self_key in self._h5file.keys():
            data = cPickle.loads(self._h5file[self.self_key].value)
            self._types = data['_types']
            self._dtypes = data['_dtypes']
        else:
            self._types = {}
            self._dtypes = {}

    def __contains__(self, key):
        if key == self.self_key:
            return False
        else:
            return self._h5file.__contains__(key)

    def __iter__(self):
        return [i for i in self._h5file if i != self.self_key].__iter__()

    def __len__(self):
        return len(self.keys() - 1)

    def keys(self):
        return [i for i in self._h5file.keys() if i != self.self_key]

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError('\'%s\' is not in the keys' % key)

        value = self._h5file[key].value
        # If it is a single string, then it is a pickled object.
        if isinstance(value, str):
            value = cPickle.loads(value)

        return value

    def __delitem__(self, key):
        if self.read_only:
            raise Exception('You cannot modify an h5dict with mode=\'r\'')
        self._types.__delitem__(key)
        self._dtypes.__delitem__(key)
        self._h5file.__delitem__(key)
        self.__self_dump__()

        if self.autoflush:
            self._h5file.flush()

    def __setitem__(self, key, value):
        if self.read_only:
            raise Exception('You cannot modify an h5dict with mode=\'r\'')
        if key == self.self_key:
            raise Exception("'%d' key is reserved by h5dict" % self.self_key)
        if not isinstance(key, str) and not isinstance(key, unicode):
            raise Exception('h5dict only accepts string keys')
        if key in self.keys():
            self.__delitem__(key)

        if issubclass(value.__class__, np.ndarray):
            self._h5file.create_dataset(name=key, data=value,
                                        compression='lzf',
                                        chunks=True)
            self._types[key] = type(value)
            self._dtypes[key] = value.dtype
        else:
            self._h5file.create_dataset(name=key,
                                        data=cPickle.dumps(value, protocol=0))
            self._types[key] = type(value)
            self._dtypes[key] = None

        self.__self_dump__()

        if self.autoflush:
            self._h5file.flush()

    def value_type(self, key):
        return self._types[key]

    def value_dtype(self, key):
        return self._dtypes[key]

    def __del__(self):
        self.flush()
        self._h5file.close()
        if self.is_tmp:
            os.remove(self.path)

    def pop(self, key):
        value = self.__getitem__(key)
        self.__delitem__(key)
        return value

    def update(self, other=None, **kwargs):
        if self.read_only:
            raise Exception('You cannot modify an h5dict with mode=\'r\'')
        if hasattr(other, 'keys'):
            for i in other:
                self.__setitem__(i, other[i])
        elif other:
            for (k, v) in other:
                self.__setitem__(k, v)
        for i in kwargs:
            self.__setitem__(i, kwargs[i])

    def flush(self):
        self._h5file.flush()

    def array_keys(self):
        return [i for i in self._h5file.keys()
                if i != self.self_key and
                issubclass(self._types[i], np.ndarray)]

    def get_dataset(self, key):
        if key not in self.array_keys():
            log.warning('The requested key {0} is not an array'.format(
                key))
        return self._h5file[key]

    def add_empty_dataset(self, key, shape, dtype):
        if self.read_only:
            raise Exception('You cannot modify an h5dict with mode=\'r\'')
        if key == self.self_key:
            raise Exception("'%d' key is reserved by h5dict" % self.self_key)
        if not isinstance(key, str) and not isinstance(key, unicode):
            raise Exception('h5dict only accepts string keys')
        if key in self.keys():
            self.__delitem__(key)

        self._h5file.create_dataset(name=key, shape=shape, dtype=dtype,
                                    compression='lzf',
                                    chunks=True)
        self._types[key] = np.ndarray
        self._dtypes[key] = dtype
