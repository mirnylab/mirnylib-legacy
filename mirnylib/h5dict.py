# (c) 2012 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Anton Goloborodko (golobor@mit.edu),
# Maksim Imakaev (imakaev@mit.edu)


"""
h5dict - HDF5-based persistent dict
===================================
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import pickle
import tempfile
import os
import collections
import logging

import h5py

log = logging.getLogger(__name__)

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


class h5dict(collections.MutableMapping):
    self_key = str('_self_key')

    def __init__(self, path=None, mode='a', autoflush=True, in_memory=False):
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
            to the disk. If path is supplied, the dict is update with data from
            supplied location.
        '''
        self.read_only = (mode == 'r')
        self.newDsetArgDict = {"compression":"lzf"}

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
            try:
                self._h5file = h5py.File(self.path, mode)
            except Exception as inst:
                print(('The file {0} is damaged or is used by other h5dict object.').format(self.path))
                raise inst

            self.__self_load__()
            self.autoflush = autoflush


    def __self_dump__(self):
        if self.self_key in list(self._h5file.keys()):
            self._h5file.__delitem__(self.self_key)

        data = {'_types': self._types, '_dtypes': self._dtypes}
        dsetData = pickle.dumps(data, protocol=-1)
        self._h5file.create_dataset(name=self.self_key, data=np.array(dsetData))

    def __self_load__(self):
        if self.self_key in list(self._h5file.keys()):
            data = pickle.loads(self._h5file[self.self_key].value)
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
        return len(list(self.keys()) - 1)

    def keys(self):
        return [i for i in list(self._h5file.keys()) if i != self.self_key]

    def __getitem__(self, key):

        if isinstance(key, six.string_types):
            key = str(key)

        if key not in list(self._h5file.keys()):

            raise KeyError('\'%s\' is not in the keys' % key)

        value = self._h5file[key].value

        # If it is a single string, then it is a pickled object.
        if "pickle" in self._h5file[key].attrs:
            try:
                value = pickle.loads(value)
            except UnicodeDecodeError:
                value = pickle.loads(value, encoding='bytes')
            except:
                raise Exception('Can\'t unpickle!')
        elif (self._h5file[key].shape == () ) and (self._h5file[key].dtype.kind in ["S", "O"]):  # old convension
            try:
                value = pickle.loads(value)
            except UnicodeDecodeError:
                value = pickle.loads(value, encoding='bytes')
            except:
                raise Exception('Can\'t unpickle!')
                

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
        if key in list(self.keys()):
            self.__delitem__(key)
        if isinstance(key, six.string_types):
            key = str(key)

        if issubclass(value.__class__, np.ndarray):
            self._h5file.create_dataset(name=key, data=value,
                                        chunks=True,
                                        **self.newDsetArgDict
                                        )
            self._types[key] = type(value)
            self._dtypes[key] = value.dtype
        else:
            self._h5file.create_dataset(name=key,data=np.array(pickle.dumps(value, protocol=-1)))

            self._h5file[key].attrs["pickle"] = True
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

    def setCompression(self, mode="gzip", compression_opts=None):
        """
        Use this to set gzip compression (or any others if available).
        Default is use gzip -4; but if you want higher compression, set compression_opts to higher values (6-7).
        """
        if mode is None:
            if "compression" in self.newDsetArgDict:
                self.newDsetArgDict.pop("compression")
            if "compression_opts" in self.newDsetArgDict:
                self.newDsetArgDict.pop("compression_opts")
            return
        self.newDsetArgDict["compression"] = mode
        if compression_opts is not None:
            self.newDsetArgDict["compression_opts"] = compression_opts


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

    def rename(self, oldkey, newkey):
        if oldkey not in self:
            raise ValueError("dataset {0} does not exist".format(oldkey))
        if newkey in self:
            raise ValueError("dataset {0} already exists".format(newkey))
        self._h5file[newkey] = self._h5file[oldkey]
        self._types[newkey] = self._types[oldkey]
        self._dtypes[newkey] = self._dtypes[oldkey]
        self.__delitem__(oldkey)
        if self.autoflush:
            self._h5file.flush()



    def flush(self):
        self._h5file.flush()

    def array_keys(self):
        return [i for i in list(self._h5file.keys())
                if i != self.self_key and
                issubclass(self._types[i], np.ndarray)]

    def get_dataset(self, key):
        dataset = self._h5file[str(key)]
        return dataset

    def add_empty_dataset(self, key, shape, dtype, **kwargs):
        if self.read_only:
            raise Exception('You cannot modify an h5dict with mode=\'r\'')
        if key == self.self_key:
            raise Exception("'%d' key is reserved by h5dict" % self.self_key)
        if key in list(self.keys()):
            self.__delitem__(key)


        self._h5file.create_dataset(name=key, shape=shape, dtype=dtype,
                                    chunks=True, **merge_two_dicts(kwargs, self.newDsetArgDict))
        self._types[key] = np.ndarray
        self._dtypes[key] = dtype
        if self.autoflush:
            self._h5file.flush()

        return self.get_dataset(key)

