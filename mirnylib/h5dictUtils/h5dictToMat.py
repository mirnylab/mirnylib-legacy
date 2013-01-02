# Copyright (C) 2010-2012 Leonid Mirny lab (mirnylab.mit.edu)
# Code written by: Maksim Imakaev (imakaev@mit.edu)
# For questions regarding using and/or distributing this code
# please contact Leonid Mirny (leonid@mit.edu)
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Saves mirnylab.h5dict.h5dict file to a folder with different datsets.

Usage
-----

python h5dictToTxt.py in_hdf5_file out_folder

out_folder may not exist, but may not be file.
Existing files will be overwritten.

Each key of the h5dict dataset is converted to a text file.
If key is an array, a matlab-compatible array is returned.
If key is not an array, python's repr command is used
to get an exact representation of an object.
"""

from mirnylib.h5dict import h5dict
from mirnylib.numutils import generalizedDtype
import os
import sys
import numpy
from scipy.io import savemat

if len(sys.argv) != 3:
    print "Usage : python h5dictToMat.py in_h5dict_file out_folder"
    print
    print "Converts h5dict file to a bunch of mat files, or text files for non-array data"
    print 'Each key of the array is converted to a separate file'
    print "Numpy.arrays are converted to a matlab-loadable txt files"
    print "Other keys are converted using python's repr command"
    print "Usage: python h5dictToTxt h5dictFile folderName"
    print "Folder will be created if not exists"
    exit()

filename = sys.argv[1]
folder = sys.argv[2]

if not os.path.exists(filename):
    raise IOError("File not found: %s" % filename)
if os.path.isfile(folder):
    raise IOError("Supplied folder is a file! ")
if not os.path.exists(folder):
    os.mkdir(folder)

mydict = h5dict(filename, 'r')
for i in mydict.keys():
    data = mydict[i]
    savefile = os.path.join(folder, i)
    if issubclass(type(data), numpy.ndarray):
        print "saving numpy array", i, "to", savefile
        if len(data.shape) > 0:
            savemat(savefile, {i: data})
            continue

    if type(data) == str:
        datarepr = data
    else:
        datarepr = repr(data)
    print "saving data", i, "to", savefile
    with open(savefile, 'w') as f:
        f.write(datarepr)
