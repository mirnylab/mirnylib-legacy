#(c) 2012 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu),
# Anton Goloborodko (golobor@mit.edu)

import numpy as np
import warnings
import mirnylib.systemutils
from numutils_new import _arrayInArray  # @UnresolvedImport @IgnorePep8
from numutils_new import fasterBooleanIndexing  # @UnresolvedImport @IgnorePep8
from numutils_new import fakeCisImpl  # @UnresolvedImport @IgnorePep8
from numutils_new import _arraySumByArray  # @UnresolvedImport @IgnorePep8
from scipy.ndimage.filters import  gaussian_filter
from mirnylib.systemutils import  deprecate
na = np.array
import  scipy.sparse.linalg
import scipy.stats
from math import cos, log, sin, sqrt
import numutils_new
#-----------------------------
"Mathematical & programming utilities first"
#-----------------------------

"numpy-related (programming) utilities"

#----------Importing cytonised functions----------

fasterBooleanIndexing, fakeCisImpl


def generalizedDtype(inObject):
    """"returns generalized dtype of an object
    upscales precision to the system-specific precision
    (int16 -> int; int32 -> int)
    Accepts all dtype-compatible objects.
    Upscales bool to int.

    Bool -> int64
    int -> int64
    float -> float64
    complex -> complex

    """
    if type(inObject) == type:
        inObject = np.dtype(inObject)
    if issubclass(type(inObject), np.ndarray):
        inObject = inObject.dtype
    if type(inObject) is not np.dtype:
        inObject = np.array(inObject).dtype
    if np.issubdtype(inObject, np.complex):
        return np.complex
    if np.issubdtype(inObject, np.float):
        return np.double
    if np.issubdtype(inObject, np.int):
        return np.int
    if np.issubdtype(inObject, np.bool):
        return np.int

    raise ValueError("Data  type not supported")


def fastMatrixSTD(inMatrix):
    "Estimates variance of the matrix"
    inMatrix = np.asarray(inMatrix)

    if len(inMatrix.flat) < 5:
        return np.mean(inMatrix)

    if len(inMatrix.flat) < 10000:
        return  inMatrix.var()

    else:
        varvar = inMatrix.flat[::100].var() + inMatrix.flat[33::231].var()\
            + inMatrix.flat[24::242].var() + inMatrix[55::413].var()
    return np.sqrt(0.25 * varvar)


def isInteger(inputData):
    """checks if input array consists of integers"""
    try:
        c = float(inputData)
        mod = np.fmod(c, 1.)
        minmod = min(mod, 1 - mod)
        if minmod < 0.0001:
            return True
        return False
    except:
        pass

    inputData = np.asarray(inputData)
    if generalizedDtype(inputData) == np.int:
        return True

    #checking if variance of the data is significantly less than 1

    varvar = fastMatrixSTD(inputData)

    mod = np.fmod(inputData, 1)
    minmod = np.minimum(mod, 1 - mod)
    if minmod.max() < 0.00001 * min(varvar, 1):
        return True
    return False



def isSymmetric(inMatrix):
    """
    Checks if the supplied matrix is symmetric.
    """

    M = inMatrix
    varDif = np.abs(M - M.T).max()
    return varDif < fastMatrixSTD(inMatrix) * 0.0000001


def _testMatrixUtils():
    print "Testing isSymmetric"
    a = np.random.randint(0, 1000, (1000, 1000))
    assert not isSymmetric(a)
    b = a + a.T
    assert isSymmetric(b)
    b = (b ** 0.01) ** 100
    assert isSymmetric(b)
    b[0] += 0.001
    assert not isSymmetric(b)
    print "Finished testing isSymmetric, test successful"
    print
    print "Testing isInteger"
    assert isInteger(1)
    assert isInteger(True)
    assert isInteger(1.)
    assert isInteger(np.sqrt(3) ** 2)
    assert isInteger([1, 2, 3, 4, 5])
    assert isInteger(1.00000000000000000001)
    myarray = np.random.randint(0, 10000, 1000000)
    assert isInteger(myarray)
    myarray = myarray * 1.
    assert isInteger(myarray)
    myarray = (myarray ** 0.01) ** 100
    assert isInteger(myarray)
    myarray[1] += 0.01
    assert not isInteger(myarray)
    print "finished testing isInteger"
    print
    print "testing generalizedDtype"
    assert generalizedDtype([1, 2]) == np.int
    assert generalizedDtype([1., 2.]) == np.double
    assert generalizedDtype([True, False]) == np.int
    assert generalizedDtype(np.float16) == np.double
    assert generalizedDtype(np.int8) == np.int
    print "All tests finished successfully!"


def openmpSum(in_array):
    """
    Performs fast sum of an array using openmm
    """
    from scipy import weave
    a = np.asarray(in_array)
    b = np.array([1.])
    N = int(np.prod(a.shape))
    N  # PyDev warning remover
    code = r"""
    int i=0;
    double sum = 0;
    omp_set_num_threads(4);
    #pragma omp parallel for      \
      default(shared) private(i)  \
      reduction(+:sum)
        for (i=0; i<N; i++)
              sum += a[i];
    b[0] = sum;
    """
    weave.inline(code, ['a', 'N', 'b'],
                     extra_compile_args=['-march=native  -O3  -fopenmp '],
                     support_code=r"""
    #include <stdio.h>
    #include <omp.h>
    #include <math.h>""",
                 libraries=['gomp'])
    return b[0]


"Manipulation with np arrays"


def rank(x):
    "Returns rank of an array"
    tmp = np.argsort(x)
    return na(np.arange(len(x)), float).take(tmp.argsort())


def trunc(x, low=0.005, high=0.005):
    "Truncates top 'low' fraction and top 'high' fraction of an array "
    lowValue, highValue = np.percentile(x, [low * 100., (1 - high) * 100.])
    return np.clip(x, a_min=lowValue, a_max=highValue)

trunk = mirnylib.systemutils.deprecate(trunc, "trunk")


def externalMergeSort(inDataset, tempDataset, chunkSize=300000000):
    """
    An in-place merge sort to work with persistent numpy-type arrays,
    in particular, h5py datasets.

    Accepts two persistent arrays: inDataset, an array to be sorted,
    and tempDataset - temporary working copy.
    Returns the result in inDataset.
    It is a two-pass algorithm, meaning that the entire dataset will be
    read and written twice to the HDD. Which is not that bad :)
    """

    if (len(inDataset) < 10000) or (len(inDataset) < chunkSize):
        data = np.array(inDataset)
        inDataset[:] = np.sort(data)
        print "Sorted using default sort"
        return
    elif chunkSize < 5 * sqrt(len(inDataset)):
        warnings.warn("Chunk size should be big enough... you provide {0} "
                      ", upscaled to {1}".format(chunkSize, 5 * int(sqrt(len(inDataset)))))
        chunkSize = 5 * int(sqrt(len(inDataset)))

    N = len(inDataset)
    bins = range(0, N, chunkSize) + [N]
    bins = zip(bins[:-1], bins[1:])  # "big chunks" of chunkSize each
    print "Sorting using %d chunks" % (len(bins),)

    #Initial pre-sorting inDataset - tempDataset
    for start, stop in bins:
        chunk = inDataset[start:stop]
        chunk.sort()
        tempDataset[start:stop] = chunk

    #Determining smaller merge chunk sizes and positions
    #each chunk is separated into chunkSize/numChunks smaller chunks
    #Smaller chunks are called "merge chunks"
    #Further they will be merged one by one
    M = len(bins)
    mergeChunkSize = chunkSize / M
    chunkLocations = []
    for start, stop in bins:
        chunkBins = range(start, stop, mergeChunkSize) + [stop]
        chunkBins = zip(chunkBins[:-1], chunkBins[1:])
        chunkLocations.append(chunkBins[::-1])
            #first chunk is last, as we use pop() later
    outputPosition = 0  # location in the output file, inDataset now

    currentChunks = []  # a set of smaller merge-chunks for each big chunk.
    for chunk in chunkLocations:
        start, end = chunk.pop()
        currentChunks.append(tempDataset[start:end])
    maxes = [i.max() for i in currentChunks]
        #A set of maximum values of working chunks
    positions = [0 for _ in currentChunks]
        #A set of current positions in working chunks

    while True:
        chInd = np.argmin(maxes)
        # An index of a chunk that has minimum maximum value right now
        #We can't merge beyound this value,
        #we need to update this chunk before doing this

        limits = [np.searchsorted(chunk, maxes[chInd],
                                  side="right") for chunk in currentChunks]
        #Up to where can we merge each chunk now? Find it using searchsorted

        currentMerge = np.concatenate([chunk[st:ed] for chunk, st,
                                       ed in zip(currentChunks,
                                                 positions, limits)])
        #Create a current array to merge

        positions = limits  # we've merged up to here
        currentMerge.sort()  # Poor man's merge
        inDataset[outputPosition:outputPosition + len(currentMerge)
                  ] = currentMerge  # write sorted merge output here.
        outputPosition += len(currentMerge)

        if len(chunkLocations[chInd]) > 0:
            # If we still have merge-chunks in that chunk
            start, end = chunkLocations[chInd].pop()  # pull out a new chunk
            chunk = tempDataset[start:end]  # update all the values
            maxes[chInd] = chunk[-1]
            currentChunks[chInd] = chunk
            positions[chInd] = 0
        else:  # If we don't have any merge-chunks in this chunk
            currentChunks.pop(chInd)  # Remove all entrees about this chunk
            positions.pop(chInd)
            chunkLocations.pop(chInd)
            maxes.pop(chInd)

        if len(currentChunks) == 0:  # If we don't have chunks
            assert outputPosition == N  # happily exit
            return


def _testExternalSort():
    a = np.random.random(150000000)
    from mirnylib.h5dict import h5dict
    t = h5dict()
    t["1"] = a
    t["2"] = a

    print "starting sort"
    import time
    tt = time.time()
    externalMergeSort(
        t.get_dataset("1"), t.get_dataset("2"), chunkSize=30000000)
    print  "time to sort", time.time() - tt
    tt = time.time()
    dif = t.get_dataset("1")[::1000000] - np.sort(a)[::1000000]
    print "time to do regular sort:", time.time() - tt
    assert dif.sum() == 0
    print "Test finished successfully!"

#_testExternalSort()


def uniqueIndex(data):
    """Returns a binary index of unique elements in an array data.
    This method is very memory efficient, much more than np.unique!
    It grabs only 9 extra bytes per record :)
    """

    args = np.argsort(data)
    index = np.zeros(len(data), bool)
    myr = range(0, len(data), len(data) / 50 + 1) + [len(data)]
    for i in xrange(len(myr) - 1):
        start = myr[i]
        end = myr[i + 1]
        dataslice = data.take(args[start:end], axis=0)
        ind = dataslice[:-1] != dataslice[1:]
        index[args[start:end - 1]] = ind
        if end != len(data):
            if data[args[end - 1]] != data[args[end]]:
                index[args[end - 1]] = True
        else:
            index[args[end - 1]] = True

    return index


def chunkedUnique(data, chunksize=5000000, return_index=False):
    """Performs unique of a long, but repetitive array.
    Set chunksize 5-10 times longer than the number of unique elements"""

    if len(data) < chunksize * 2:
        return np.unique(data, return_index=return_index)

    mytype = data.dtype
    bins = range(0, len(data), chunksize) + [len(data)]
    bins = zip(bins[:-1], bins[1:])
    current = np.zeros(0, dtype=mytype)
    if return_index:
        currentIndex = np.zeros(0, dtype=int)
    for start, end in bins:
        chunk = data[start:end]
        if return_index:
            chunkUnique, chunkIndex = np.unique(chunk, return_index=True)
            chunkIndex += start
            current, extraIndex = np.unique(np.concatenate(
                [current, chunkUnique]), return_index=True)
            currentIndex = np.concatenate(
                [currentIndex, chunkIndex])[extraIndex]
        else:
            chunkUnique = np.unique(chunk)
            current = np.unique(np.concatenate([current, chunkUnique]))

    if return_index:
        return current, currentIndex
    else:
        return current


def trimZeros(x):
    "trims leading and trailing zeros of a 1D/2D array"
    if len(x.shape) == 1:
        nz = np.nonzero(x)[0]
        return x[nz.min():nz.max() + 1]
    ax1 = np.nonzero(np.sum(x, axis=0))[0]
    ax2 = np.nonzero(np.sum(x, axis=1))[0]
    return x[ax1.min():ax1.max() + 1, ax2.min(): ax2.max() + 1]


def zoomOut(x, shape):
    "rescales an array preserving the structure and total sum"
    M1 = shape[0]
    M2 = shape[1]
    N1 = x.shape[0]
    N2 = x.shape[1]
    if (N1 < M1) or (N2 < M2):
        d1 = M1 / N1 + 1
        d2 = M2 / N2 + 1

        newx = np.zeros((d1 * N1, d2 * N2))
        for i in xrange(d1):
            for j in xrange(d2):
                newx[i::d1, j::d2] = x / (1. * d1 * d2)
        x = newx
        N1, N2 = N1 * d1, N2 * d2  # array is bigger now

    shift1 = N1 / float(M1) + 0.000000001
    shift2 = N2 / float(M2) + 0.000000001
    x = np.asarray(x, dtype=float)
    tempres = np.zeros((M1, N2), float)
    for i in xrange(N1):
        beg = (i / shift1)
        end = ((i + 1) / shift1)
        if int(beg) == int(end):
            tempres[beg, :] += x[i, :]
        else:
            tempres[beg, :] += x[i, :] * (int(end) - beg) / (end - beg)
            tempres[beg + 1, :] += x[i, :] * (end - int(end)) / (end - beg)
    res = np.zeros((M1, M2), float)
    for i in xrange(N2):
        beg = (i / shift2)
        end = ((i + 1) / shift2)
        if int(beg) == int(end):
            res[:, beg] += tempres[:, i]
        else:
            res[:, beg] += tempres[:, i] * (int(end) - beg) / (end - beg)
            res[:, beg + 1] += tempres[:, i] * (end - int(end)) / (end - beg)
    return res


smartZoomOut = mirnylib.systemutils.deprecate(
    zoomOut, "smartZoomOut")  # backwards compatibility


def coarsegrain(array, size, extendEdge=False):
    """coarsegrains array by summing values in sizeXsize squares;
    truncates the unused squares"""
    array = np.asarray(array, dtype=generalizedDtype(array.dtype))

    if not extendEdge:
        if len(array.shape) == 2:
            N = len(array) - len(array) % size
            array = array[:N, :N]
            a = np.zeros((N / size, N / size), float)
            for i in xrange(size):
                for j in xrange(size):
                    a += array[i::size, j::size]
            return a
        if len(array.shape) == 1:
            array = array[:(len(array) / size) * size]
            narray = np.zeros(len(array) / size, float)
            for i in xrange(size):
                narray += array[i::size]
            return narray
    else:
        N = len(array)
        if len(array.shape) == 2:
            resultSize = np.ceil(float(N) / size)
            a = np.zeros((resultSize, resultSize), float)
            for i in xrange(size):
                for j in xrange(size):
                    add = array[i::size, j::size]
                    a[:add.shape[0], :add.shape[1]] += add
            return a
        if len(array.shape) == 1:
            resultSize = np.ceil(float(N) / size)
            a = np.zeros((resultSize), float)
            for i in xrange(size):
                add = array[i::size]
                a[:len(add)] += add
            return a


def partialCorrelation(x, y, z,
                       corr=lambda x, y: scipy.stats.spearmanr(x, y)[0]):
    xy, xz, yz = corr(x, y), corr(x, z), corr(y, z)
    return (xy - xz * yz) / (sqrt(1 - xz ** 2) * sqrt(1 - yz ** 2))


def robustPartialCorrelation(x, y, z, smeerSize=100,
                             corrFun=lambda x, y: scipy.stats.spearmanr(x, y)[0]):

    args = np.argsort(z)

    def smeer(array):
        "subtracts averages from bins in Z"
        aSort = array[args]
        smoothed = gaussian_filter(aSort, smeerSize)
        aSort -= smoothed
        newArray = np.zeros_like(array, dtype=float)
        newArray[args] = aSort
        return newArray
    newX = smeer(x)
    newY = smeer(y)
    return corrFun(newX, newY), (newX, newY)


"Array indexing-related utilities, written in np/c++"


def arraySearch(array, tosearch):
    """returns location of tosearch in array;
    -->> assumes that elements exist!!! <--- """
    inds = np.argsort(array)
    arSorted = array.take(inds, axis=0)
    newinds = np.searchsorted(arSorted[:-1], tosearch)
    return inds[newinds]


def arrayInArray(array, filterarray, chunkSize="auto"):
    """gives you boolean array of indices of elements in array,
    that are contained in filterarray
    a faster version of  [(i in filterarray) for i in array]
    In fact, it's faster than numpy's buildin function,
    that is written in pure numpy but uses 2 argsorts instead of one

    This method is additionaly optimized
    for large array and short filterarray!\n
    Actual implementation is in the _arrayInArray"""  # sorted array
    array = np.asarray(array)
    filterarray = np.unique(filterarray)
    if chunkSize == "auto":
        chunkSize = max(4 * len(filterarray), 50000)
    if len(array) < 2.5 * chunkSize:
        return _arrayInArray(array, filterarray)

    mask = np.zeros(len(array), 'bool')
    N = len(array)
    chunks = range(0, N, chunkSize) + [N]
    bins = zip(chunks[:-1], chunks[1:])
    for start, end in bins:
        mask[start:end] = _arrayInArray(array[start:end], filterarray)
    return mask


def _testArayInArray():
    a = np.random.randint(0, 1000000, 10000000)
    b = np.random.randint(0, 1000000, 500000)
    arrayInArray(a, b)
    import time
    tt = time.time()
    res1 = arrayInArray(a, b)
    print "optimized way: ", time.time() - tt
    tt = time.time()
    res2 = _arrayInArray(a, b)
    print "standard way: ", time.time() - tt
    tt = time.time()
    res3 = np.in1d(a, b)
    print "np way: ", time.time() - tt
    assert (res1 != res2).sum() == 0
    assert (res1 != res3).sum() == 0
    print "arrayInArray test finished successfully "


def arraySumByArray(array, filterarray, meanarray):
    """faster [sum(meanarray[array == i]) for i in filterarray]
    Current method is a wrapper that optimizes
    this method for speed and memory efficiency.
    """
    if (len(array) / len(filterarray) > 4) and (len(array) > 20000000):
        M = len(array) / len(filterarray) + 1
        chunkSize = min(len(filterarray) * M, 10000000)
        bins = range(0, len(array), chunkSize) + [len(array)]
        toreturn = np.zeros(len(filterarray), float)
        for i in xrange(len(bins) - 1):
            toreturn += _arraySumByArray(array[bins[i]:bins[i + 1]],
                                         filterarray,
                                         meanarray[bins[i]:bins[i + 1]])
        return toreturn
    else:
        return _arraySumByArray(array, filterarray)


def _testArraySumByArray():
    a = np.random.randint(0, 1000000, 30000000)
    b = np.random.randint(0, 1000000, 500000)
    c = np.random.random(30000000)

    r1 = _arraySumByArray(a, b, c)
    r2 = arraySumByArray(a, b, c)

    dif = (r1 - r2).sum()
    print "Difference is {0}, should be less than 1e-10".format(dif)
    assert dif < 1e-10

#_testArraySumByArray()


def _sumByArray(array, filterarray, dtype="int64"):
    "actual implementation of sumByArray"
    arsort = np.sort(array)
    diffs = np.r_[0, np.nonzero(np.diff(arsort) > 0.5)[0] + 1, len(arsort)]
    if dtype is not None:
        diffs = np.array(diffs, dtype=dtype)
    values = arsort.take(diffs[:-1])
    del arsort
    allinds = np.searchsorted(values[:-1], filterarray)
    notexist = values.take(allinds) != filterarray
    del values
    c = diffs.take(allinds + 1) - diffs[allinds]
    c[notexist] = 0
    return c


def sumByArray(array, filterarray, dtype="int64"):
    """faster [sum(array == i) for i in filterarray]
    Current method is a wrapper that
    optimizes this method for speed and memory efficiency.
    """
    if (len(array) / len(filterarray) > 2) and (len(array) > 20000000):
        M = len(array) / len(filterarray) + 1
        chunkSize = min(len(filterarray) * M, 10000000)
        bins = range(0, len(array), chunkSize) + [len(array)]
        toreturn = np.zeros(len(filterarray), array.dtype)
        for i in xrange(len(bins) - 1):
            toreturn += _sumByArray(
                array[bins[i]:bins[i + 1]], filterarray, dtype)
        return toreturn
    else:
        return _sumByArray(array, filterarray, dtype)


"Mathematical utilities"


def corr2d(x):
    "FFT-based 2D correlation"
    x = np.array(x)
    t = np.fft.fft2(x)
    return np.real(np.fft.ifft2(t * np.conjugate(t)))


def logbins(a, b, pace=0, N_in=0):
    """Create log-spaced bins.

    Parameters
    ----------
    a, b : int
        the range of the bins
    pace : float
        approximate multiplier for each bin. Specify this or `N_in`.
    N_in : int
        number of bins. Specify this or `pace`.

    """
    a = int(a)
    b = int(b)
    beg = log(a)
    end = log(b - 1)
    if pace != 0:
        pace = log(pace)
        N = int((end - beg) / pace)
        pace = (end - beg) / N
    elif N_in != 0:
        N = N_in
        pace = (end - beg) / N
    else:
        raise ValueError("Specify either pace or N_in")
    if N > (b - a):
        raise ValueError("Cannot create more bins than elements")
    mas = np.arange(beg, end + 0.000000001, pace)
    ret = np.exp(mas)
    surpass = np.arange(a, a + N)
    replace = surpass > ret[:N] - 1
    ret[replace] = surpass
    ret = np.array(ret, dtype=np.int)
    ret[-1] = b
    return list(ret)


def rescale(data):
    "rescales array to zero mean unit variance"
    data = np.asarray(data, dtype=float)
    return (data - data.mean()) / sqrt(data.var())


def autocorr(x):
    "autocorrelation function"
    x = rescale(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size / 2:]


def rotationMatrix(theta):
    "Calculates 3D rotation matrix based on angles"
    tx, ty, tz = theta
    Rx = np.array([[1, 0, 0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
    Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))


def random_on_sphere(r=1):
    while True:
        a = np.random.random(2)
        x1 = 2 * a[0] - 1
        x2 = 2 * a[1] - 1
        if x1 ** 2 + x2 ** 2 > 1:
            continue
        t = sqrt(1 - x1 ** 2 - x2 ** 2)
        x = r * 2 * x1 * t
        y = r * 2 * x2 * t
        z = r * (1 - 2 * (x1 ** 2 + x2 ** 2))
        return (x, y, z)


def random_in_sphere(r=1):
    while True:
        a = np.random.random(3) * 2 - 1
        if np.sum(a ** 2) < 1:
            return r * a
randomInSphere = random_in_sphere
randomOnSphere = random_on_sphere


#---------------------------------------------------------
"Iterative correction, PCA and other Hi-C related things"
#---------------------------------------------------------

#-------------Importing cytonized functions--------------

observedOverExpected = \
    numutils_new.observedOverExpected  # @UndefinedVariable @IgnorePep8
ultracorrectSymmetricByMask = \
    numutils_new.ultracorrectSymmetricByMask  # @UndefinedVariable @IgnorePep8
ultracorrectSymmetricWithVector = \
    numutils_new.ultracorrectSymmetricWithVector  # @UndefinedVariable @IgnorePep8


def adaptiveSmoothing(matrix, parameter, alpha=0.5,
                      mask="auto", originalCounts="matrix"):
    """This is a super-cool method to smooth a heatmap.
    Smoothes each point into a gaussian, encoumpassing parameter
    raw reads, taked from originalCounts data, or from matrix if not provided
    """
    matrix = np.asarray(matrix, dtype=np.double)
    if originalCounts == "matrix":
        originalCounts = matrix

    if mask is None:
        mask = np.ones(matrix.shape, bool)

    elif mask == "auto":
        sum1 = np.sum(matrix, axis=0) > 0
        sum0 = np.sum(matrix, axis=1) > 0
        mask = sum0[:, None] * sum1[None, :]

    mask = np.array(mask, dtype=float)
    antimask = np.nonzero(mask.flat == False)[0]
    assert mask.shape == matrix.shape

    #Getting sure that there are no reads in masked regions
    originalCounts.flat[antimask] = 0
    matrix.flat[antimask] = 0

    def coolFilter(matrix, value):
        "gaussian filter for masked data"
        mf = gaussian_filter(mask, value)
        mf.flat[antimask] = 1
        #mfDivided = matrix / mf
        #mfDivided.flat[antimask] = 0
        gf = gaussian_filter(matrix, value)
        gf.flat[antimask] = 0
        return gf / mf

    outMatrix = np.zeros_like(matrix)
    nonZero = matrix > 0
    nonZeroSum = nonZero.sum()
    values = np.r_[np.logspace(-1, 4, 50, 2)]
    values = values[values * 2 * np.pi > 1]
    #print values
    covered = np.zeros(matrix.shape, dtype=bool)
    #outMatrix[covered] += matrix[covered]

    for value in values:
        #finding normalization of a discrete gaussian filter
        test = np.zeros((8 * value, 8 * value), dtype=float)
        test[4 * value, 4 * value] = 1
        stest = gaussian_filter(test, value)
        stest /= stest[4 * value, 4 * value]
        norm = stest.sum()

        smoothed = gaussian_filter(originalCounts, value) * norm
        smoothed -= alpha * originalCounts
        #assert smoothed.min() >= -1e-10

        #Indeces to smooth on that iteration
        new = (smoothed > parameter) * (covered != True) * nonZero
        newInds = np.nonzero(new.flat)[0]

        #newReads = originalCounts.flat[newInds]
        #directPercent = (newReads - p) / (1. * p)
        #directPercent[newReads <= p] = 0
        #directPercent[newReads >= 2 * p] = 1

        newMatrix = np.zeros_like(matrix, dtype=np.double)
        newMatrix.flat[newInds] = matrix.flat[newInds]
        #newMatrix.flat[newInds] = matrix.flat[newInds] * (1. - directPercent)
        #outMatrix.flat[newInds] += matrix.flat[newInds] * directPercent

        outMatrix += coolFilter(newMatrix, value)
        covered[new] = True

        if covered.sum() == nonZeroSum:
            break
    #print matrix.sum(), outMatrix.sum()
    return outMatrix


def maskPCA(A, mask):
    "attempts to perform PCA-like analysis of an array with a masked part"
    from numpy import linalg
    mask = np.array(mask, int)
    bmask = mask == 0
    A[bmask] = 0
    sums = np.sum(A, axis=0)
    means = sums / (1. * np.sum(mask, axis=0))
    M = (A - means).T
    M[bmask] = 0
    covs = np.zeros((len(M), len(M)), float)
    for i in xrange(len(M)):
        myvector = M[i]
        mymask = mask[i]
        allmask = mask * mymask[None, :]
        tocov = myvector[None, :] * M
        covsums = tocov.sum(axis=1)
        masksums = allmask.sum(axis=1)
        covs[i] = covsums / masksums
    [latent, coeff] = linalg.eig(covs)
    print latent[:4]
    return coeff


def PCA(A, numPCs=6):
    """performs PCA analysis, and returns 6 best principal components
    result[0] is the first PC, etc"""
    A = np.array(A, float)
    M = (A - np.mean(A.T, axis=1)).T
    covM = np.dot(M, M.T)
    [latent, coeff] = scipy.sparse.linalg.eigsh(covM, numPCs)
    print "Eigenvalues are:", latent
    return (np.transpose(coeff[:, ::-1]), latent[::-1])


def EIG(A, numPCs=3):
    """Performs mean-centered engenvector expansion
    result[0] is the first EV, etc.;
    by default returns 3 EV
    """
    A = np.array(A, float)
    M = (A - np.mean(A))  # subtract the mean (along columns)
    if isSymmetric(A):
        [latent, coeff] = scipy.sparse.linalg.eigsh(M, numPCs)
    else:
        [latent, coeff] = scipy.sparse.linalg.eigs(M, numPCs)
    alatent = np.argsort(np.abs(latent))
    print "eigenvalues are:", latent[alatent]
    coeff = coeff[:, alatent]
    return (np.transpose(coeff[:, ::-1]), latent[alatent][::-1])


def project(data, vector):
    "project data on a single vector"
    dot = np.sum((data * vector[:, None]), axis=0)
    den = (vector * vector).sum()
    return vector[:, None] * (dot / den)[None, :]


def projectOnEigenvectors(data, N=1, forceSymmetrize=False):
    "projects symmetric data on the first N eigenvalues of mean-centered data"

    data = np.asarray(data)
    if forceSymmetrize:
        data = 0.5 (data + data.T)
    if not isSymmetric(data):
        raise ValueError("projections on eigenvectors only works for symmetric data")

    meanOfData = np.mean(data)
    vectors, values = EIG(data, N)

    ndata = meanOfData
    for i in xrange(N):
        ndata += values[i] * vectors[i][:, None] * vectors[i][None, :]
    return ndata

projectOnEigenvalues = deprecate(
    projectOnEigenvectors, "projectOnEigenvalues")


def _testProjectOnEigenvectors():
    "for a smoothed matrix last eigenvector is essentially flat"
    print "Testing projection on eigenvectors"
    a = np.random.random((100, 100))
    a = gaussian_filter(a, 3)
    sa = a + a.T
    assert np.max(np.abs((projectOnEigenvectors(sa, 99) - sa))) < 0.00001
    print "Test finished successfully!"


def correct(y):
    "Correct non-symmetric or symmetirc data once"
    x = np.array(y, float)
    s = np.sum(x, axis=1)
    s /= np.mean(s[s != 0])
    s[s == 0] = 1
    s2 = np.sum(x, axis=0)
    s2 /= np.mean(s2[s2 != 0])
    s2[s2 == 0] = 1
    return x / (s2[None, :] * s[:, None])


def correctInPlace(x):
    "works for non-symmetric and symmetric data"
    s = np.sum(x, axis=1)
    s /= np.mean(s[s != 0])
    s[s == 0] = 1
    s2 = np.sum(x, axis=0)
    s2 /= np.mean(s2[s2 != 0])
    s2[s2 == 0] = 1
    x /= (s2[None, :] * s[:, None])


def ultracorrect(x, M=20):
    "just iterative correction of symmetric matrix"
    x = np.array(x, float)
    if x.sum() == 0:
        return x
    print np.mean(x),
    newx = np.array(x)
    for _ in xrange(M):
        correctInPlace(newx)
    print np.mean(newx),
    newx /= (1. * np.mean(newx) / np.mean(x))
    print np.mean(newx)
    return newx


def correctBias(y):
    "performs single correction and returns data + bias"
    x = np.asarray(y, dtype=float)
    s = np.sum(x, axis=1)
    s /= np.mean(s[s != 0])
    s[s == 0] = 1
    return x / (s[None, :] * s[:, None]), s


def ultracorrectBiasReturn(x, M=20):
    "performs iterative correction and returns bias"
    x = np.array(x, float)
    print np.mean(x),
    newx = np.array(x)
    ball = np.ones(len(x), float)
    for _ in xrange(M):
        newx, b = correctBias(newx)
        ball *= b
    print np.mean(newx),
    newx /= (1. * np.mean(newx) / np.mean(x))
    print np.mean(newx)
    return newx, ball


def fillDiagonal(inArray, diag, offset=0):
    "Puts diag in the offset's diagonal of inArray"
    N = inArray.shape[0]
    assert inArray.shape[1] == N
    if offset >= 0:
        inArray.flat[offset:N * (N - offset):N + 1] = diag
    else:
        inArray.flat[(-offset * N)::N + 1] = diag


def removeDiagonals(inArray, m):
    """removes up to mth diagonal in array
    m = 0: main
    m = 1: 3 diagonals, etc.
    """
    for i in xrange(-m, m + 1):
        fillDiagonal(inArray, 0, i)


def shuffleAlongDiagonal(inMatrix):
    "Shuffles inMatrix along each diagonal"
    assert len(inMatrix.shape) == 2
    sh = inMatrix.shape
    assert sh[0] == sh[1]
    N = sh[0]
    for i in xrange(-N + 1, N):
        diag = np.diagonal(inMatrix, i).copy()
        np.random.shuffle(diag)
        fillDiagonal(inMatrix, diag, i)
    return inMatrix


def smeerAlongDiagonal(inMatrix):
    assert len(inMatrix.shape) == 2
    sh = inMatrix.shape
    assert sh[0] == sh[1]
    N = sh[0]
    for i in xrange(-N + 1, N):
        diag = np.diagonal(inMatrix, i).mean()
        fillDiagonal(inMatrix, diag, i)
    return inMatrix


def create_regions(a):
    """creates array of start/stop positions
    of continuous nonzero regions of array a"""
    a = np.array(a, int)
    a = np.concatenate([np.array([0], int), a, np.array([0], int)])
    a1 = np.nonzero(a[1:] * (1 - a[:-1]))[0]
    a2 = np.nonzero(a[:-1] * (1 - a[1:]))[0]
    return np.transpose(np.array([a1, a2]))


def _test():
    _testArayInArray()
    _testExternalSort()
    _testArraySumByArray()
    _testMatrixUtils()
    _testProjectOnEigenvectors()
