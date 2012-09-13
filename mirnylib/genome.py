# Copyright (C) 2010-2012 Leonid Mirny lab (mirnylab.mit.edu)
# Code written by: Anton Goloborodko (golobor@mit.edu),
# Maksim Imakaev (imakaev@mit.edu)
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


'''A Genome object contains the cached properties of a genome.

Glossary
--------

positional identifier : a number that can be decomposed into a tuple
    (chromosome index, base pair).

chromosome string label : the name of a chromosome.
    Examples: 'X', 'Y', '1', '22'.

chromosome index : a zero-based numeric index of a chromosome.
    For numbered chromosomes it is int(label) - 1, unless some of the
    chromosomes are absent. The chromosomes 'X', 'Y', 'M' are indexed
    after, the rest is indexed in alphabetical order.

concatenated genome : a genome with all chromosomes merged together
into one sequence.

binned genome : a genome splitted into bins `resolution` bp each.

binned concatenated genome : a genome with chromosomes binned and merged.
    GOTCHA: since the genome is binned FIRST and merged after that, the
    number of bins may be greater than (sum of lengths / resolution).
    The reason for this behavior is that the last bin in a chromosome
    is usually shorter than `resolution` but still counts as a full bin.
'''

import os
import glob
import re
import numpy
import warnings
import Bio.SeqIO
import Bio.SeqUtils
import Bio.Restriction
Bio.Restriction  # To shut up Eclipse warning
import joblib
from scipy import weave
import logging
import numutils

log = logging.getLogger(__name__)

class Genome(object):
    """A class to compute and cache various properties of genomic sequences."""

    def _memoize(self, func_name):
        '''Local version of joblib memoization.
        The key difference is that _memoize() takes into account only the
        relevant attributes of a Genome object (folder, name, gapFile,
        chrmFileTemplate) and ignores the rest.

        The drawback is that _memoize() doesn't check for the changes in the
        code of the function!'''
        if not hasattr(self, '_mymem'):
            self._mymem = joblib.Memory(cachedir=self.genomePath)

        def run_func(readChrms, gapFile, chrmFileTemplate,
                     func_name, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

        mem_func = self._mymem.cache(run_func)

        def memoized_func(*args, **kwargs):
            return mem_func(
                self.readChrms, self.gapFile,
                self.chrmFileTemplate, func_name, *args, **kwargs)

        return memoized_func

    def _parseGapFile(self):
        """Parse a .gap file to determine centromere positions.
        """
        gapPath = os.path.join(self.genomePath, self.gapFile)
        if not os.path.isfile(gapPath):
            log.warning(
                'Gap file not found!\n'
                'Please provide a link to a gapfile or '
                'put a file gap.txt in a genome directory')
            return

        gapFile = open(gapPath).readlines()

        self.cntrStarts = -1 * numpy.ones(self.chrmCount, int)
        self.cntrEnds = -1 * numpy.zeros(self.chrmCount, int)

        for line in gapFile:
            splitline = line.split()
            if splitline[7] == 'centromere':
                chrm_str = splitline[1][3:]
                if chrm_str in self.label2idx:
                    chrm_idx = self.label2idx[chrm_str]
                    self.cntrStarts[chrm_idx] = int(splitline[2])
                    self.cntrEnds[chrm_idx] = int(splitline[3])

        self.cntrMids = (self.cntrStarts + self.cntrEnds) / 2
        lowarms = numpy.array(self.cntrStarts)
        higharms = numpy.array(self.chrmLens) - numpy.array(self.cntrEnds)
        self.maxChrmArm = max(lowarms.max(), higharms.max())

    def _extractChrmLabel(self, fastaName):
        # First assume a whole filename as input (e.g. 'chr01.fa')
        _, fastaName = os.path.split(fastaName)
        regexp = self.chrmFileTemplate % ('(.*)')
        search_results = re.search(regexp, fastaName)
        # If not, assume that only the name is supplied as input (e.g. 'chr01')
        if search_results is None:
            regexp = self.chrmFileTemplate.split('.')[0] % ('(.*)')
            search_results = re.search(regexp, fastaName)
        chrm_label = search_results.group(1)

        # Remove leading zeroes.
        if chrm_label.isdigit():
            chrm_label = str(int(chrm_label))

        return chrm_label

    def _scanGenomeFolder(self):
        if not os.path.isdir(self.genomePath):
            raise Exception('{0} is not a folder'.format(self.genomePath))

        self.fastaNames = [os.path.join(self.genomePath, i)
                           for i in glob.glob(os.path.join(
                                          self.genomePath,
                                          self.chrmFileTemplate % ('*',)))]
        log.debug('Scan genome folder: {0}'.format(self.genomePath))
        log.debug('FASTA files are found: {0}'.format(self.fastaNames))

        if len(self.fastaNames) == 0:
            raise Exception('No Genome files found at %s' % self.genomePath)

        # Read chromosome IDs.
        self.chrmLabels = []
        filteredFastaNames = []
        for i in self.fastaNames:
            chrm = self._extractChrmLabel(i)
            if (not(self.readChrms)
                or
                (chrm.isdigit() and '#' in self.readChrms)
                or
                chrm in self.readChrms):

                self.chrmLabels.append(chrm)
                filteredFastaNames.append(i)
                log.debug('Convert %s FASTA filename to %s chromosome label, '
                              'store in the Genome object', i, chrm)
            else:
                log.debug('Convert %s FASTA filename to %s chromosome label, '
                              'discard', i, chrm)

        self.fastaNames = filteredFastaNames
        log.debug('The following FASTA files satisfy the readChrms variable '
                      '(={0}): {1}'.format(self.readChrms, self.fastaNames))

        if len(self.fastaNames) == 0:
            raise Exception('No Genome files at %s contain '
                            'the specified chromosomes' % self.genomePath)

        # Convert IDs to indices:
        # A. Convert numerical IDs.
        num_ids = [i for i in self.chrmLabels if i.isdigit()]
        log.debug('The chromosomes with numerical IDs: {0}'.format(num_ids))
        # Sort IDs naturally, i.e. place '2' before '10'.
        num_ids.sort(key=lambda x: int(re.findall(r'\d+$', x)[0]))

        self.chrmCount = len(num_ids)
        self.label2idx = dict(
            [(num_ids[i], int(i)) for i in xrange(len(num_ids))])
        self.idx2label = dict(
            [(int(i), num_ids[i]) for i in xrange(len(num_ids))])

        # B. Convert non-numerical IDs. Give the priority to XYM over the rest.
        nonnum_ids = [i for i in self.chrmLabels if not i.isdigit()]
        log.debug('The chromosomes with non-numerical IDs: {0}'.format(nonnum_ids))
        for i in ['M', 'Y', 'X']:
            if i in nonnum_ids:
                nonnum_ids.pop(nonnum_ids.index(i))
                nonnum_ids.insert(0, i)

        for i in nonnum_ids:
            self.label2idx[i] = self.chrmCount
            self.idx2label[self.chrmCount] = i
            self.chrmCount += 1

        # Sort fastaNames and self.chrmLabels according to the indices:
        self.chrmLabels = zip(*sorted(self.idx2label.items(),
                                      key=lambda x: x[0]))[1]
        self.fastaNames.sort(
            key=lambda path: self.label2idx[self._extractChrmLabel(path)])
        log.debug('The genome folder is scanned successfully.')

    def __init__(self, genomePath, gapFile='gap.txt',
                 chrmFileTemplate='chr%s.fa',
                 readChrms=['#', 'X', 'Y', 'M']):
        '''
        A class that stores cached properties of a genome. To initialize,
        a Genome object needs FASTA files with chromosome sequences.
        For the basic definitions please refer to the module documentation.

        Parameters
        ----------
        (for the constructor method)

        genomePath : str
            The path to the folder with the FASTA files.

        gapFile : str
            The path to the gap file relative to genomePath.

        chrmFileTemplate : str
            The template of the FASTA file names.

        readChrms : list of str
            The list with the string labels of chromosomes to read from the
            genome folder. '#' stands for chromosomes with numerical labels
            (e.g. 1-22 for human). If readChrms is empty then read all
            chromosomes.

        Attributes
        ----------

        chrmCount : int
            the total number of chromosomes.

        chrmLabels : list of str
            a list of chromosomal IDs sorted in ascending index order.

        fastaNames : list of str
            FASTA files for sorted in ascending index order of respective
            chromosomes.

        genomePath : str
            The path to the folder with the genome.

        name : str
            The string identifier of the genome, the name of the last folder in
            the path.

        label2idx : dict
            a dictionary for conversion between string chromosome labels and
            zero-based indices.

        idx2label : dict
            a dictionary for conversion between zero-based indices and
            string chromosome labels.

        seqs : list of str
            a list of chromosome sequences. Loads on demand.

        chrmLens : list of int
            The lengths of chromosomes.

        maxChrmLen : int
            The length of the longest chromosome.

        cntrStarts : array of int
            The start positions of the centromeres.

        cntrMids : array of int
            The middle positions of the centromeres.

        cntrEnds : array of int
            The end positions of the centromeres.

        The following attributes are calculated after setResolution() is called:

        resolution : int
            The size of a bin for the binned values.

        chrmLensBin : array of int
            The lengths of chromosomes in bins.

        chrmStartsBinCont : array of int
            The positions of the first bins of the chromosomes in the
            concatenated genome.

        chrmEndsBinCont : array of int
            The positions of the last plus one bins of the chromosomes in the
            concatenated genome.

        chrmIdxBinCont : array of int
            The index of a chromosome in each bin of the concatenated genome.

        posBinCont : array of int
            The index of the first base pair in a bin in the concatenated
            genome.

        cntrMidsBinCont : array of int
            The position of the middle bin of a centromere in the concatenated
            genome.

        chrmArmLimitsBinCont: array of int
            The position of the chromosome arm borders in the concatenated
            genome.

        GCBin : list of arrays of float
            % of GC content of bins in individual chromosomes.

        unmappedBasesBin : list of arrays of int
            Number of bases with N's for each bin

        mappedBasesBin : list of arrays of int
            Number of sequenced bases for each bin

        binSizesbp : list of arrays of int
            Size of each bin. Is less than *resolution* for the last bin only.

        The following attributes are calculated after setEnzyme() is called:

        enzymeName : str
            The restriction enzyme used to find the restriction sites.

        rsites : list of arrays of int
            The indices of the first base pairs of restriction fragments
            in individual chromosomes.

        rfragMids : list of arrays of int
            The indices of the middle base pairs of restriction fragments
            in individual chromosomes.

        rsiteIds : array of int
            The position identifiers of the first base pairs of restriction
            fragments.

        rsiteMidIds : array of int
            The position identifiers of the middle base pairs of restriction
            fragments.

        rsiteChrms : array of int
            The indices of chromosomes for restriction sites in corresponding
            positions of rsiteIds and rsiteMidIds.
        '''
        # Set the main attributes of the class.
        self.genomePath = os.path.abspath(genomePath)
        self.folderName = os.path.split(self.genomePath)[-1]
        self.readChrms = set(readChrms)
        self.gapFile = gapFile
        self.chrmFileTemplate = chrmFileTemplate

        log.debug('Initialize a Genome object genomePath=%s, readChrms=%s, '
                    'gapFile=%s, chrmFileTemplate=%s', self.genomePath,
                    self.readChrms, self.gapFile, self.chrmFileTemplate)

        # Scan the folder and obtain the list of chromosomes.
        self._scanGenomeFolder()

        # Get the lengths of the chromosomes.
        self.chrmLens = self.getChrmLen()
        self.maxChrmLen = max(self.chrmLens)
        # FragIDmult is used in (chrm, frag) -> fragID conversion.
        self.fragIDmult = self.maxChrmLen + 1000

        # Parse a gap file and mark the centromere positions.
        self._parseGapFile()

    def getChrmLen(self):
        # At the first call redirects itself to a memoized private function.
        self.getChrmLen = self._memoize('_getChrmLen')
        return self.getChrmLen()

    def _getChrmLen(self):
        return numpy.array([len(self.seqs[i])
                            for i in xrange(0, self.chrmCount)])

    def getGCBin(self, resolution):
        # At the first call the function rewrites itself with a memoized
        # private function.
        self.getGCBin = self._memoize('_getGCBin')
        return self.getGCBin(resolution)

    def _getGCBin(self, resolution):
        GCBin = []
        for chrm in xrange(self.chrmCount):
            chrmSizeBin = int(self.chrmLens[chrm] // resolution) + 1
            GCBin.append(numpy.ones(chrmSizeBin, dtype=numpy.float))
            for j in xrange(chrmSizeBin):
                GCBin[chrm][j] = self.getGC(
                    chrm, j * int(resolution), (j + 1) * int(resolution))
        return GCBin

    def getUnmappedBasesBin(self, resolution):
        # At the first call the function rewrites itself with a memoized
        # private function.
        self.getUnmappedBasesBin = self._memoize('_getUnmappedBasesBin')
        return self.getUnmappedBasesBin(resolution)

    def _getUnmappedBasesBin(self, resolution):
        unmappedBasesBin = []
        for chrm in xrange(self.chrmCount):
            chrmSizeBin = int(self.chrmLens[chrm] // resolution) + 1
            unmappedBasesBin.append(numpy.ones(chrmSizeBin, dtype=numpy.int))
            for j in xrange(chrmSizeBin):
                unmappedBasesBin[chrm][j] = self.getUnmappedBases(
                    chrm, j * int(resolution), (j + 1) * int(resolution))
        return unmappedBasesBin

    def getRsites(self, enzymeName):
        # At the first call redirects itself to a memoized private function.
        self.getRsites = self._memoize('_getRsites')
        return self.getRsites(enzymeName)

    def _getRsites(self, enzymeName):
        '''Returns: tuple(rsites, rfrags)
        Finds restriction sites and mids of rfrags for a given enzyme
        '''

        #Memorized function
        enzymeSearchFunc = eval('Bio.Restriction.%s.search' % enzymeName)
        rsites = []
        rfragMids = []
        for i in xrange(self.chrmCount):
            rsites.append(numpy.r_[
                0, numpy.array(enzymeSearchFunc(self.seqs[i].seq)) + 1,
                len(self.seqs[i].seq)])
            rfragMids.append((rsites[i][:-1] + rsites[i][1:]) / 2)

        # Remove the first trivial restriction site (0)
        # to equalize the number of points in rsites and rfragMids.
        for i in xrange(len(rsites)):
            rsites[i] = rsites[i][1:]

        return rsites, rfragMids

    @property
    def seqs(self):
        if not hasattr(self, "_seqs"):
            self._seqs = []
            for i in xrange(self.chrmCount):
                self._seqs.append(Bio.SeqIO.read(open(self.fastaNames[i]),
                                                 'fasta'))
        return self._seqs

    def createGapFile(self, centromere_positions):
        """Create a gap file with the centromere positions.

        Use this method, if the genome you're using has no gap file.

        Parameters
        ----------

        centromere_positions : dict of (int, int)
            A dictionary with centromere positions. The keys are the chromosome
            string labels and the values are (centromereStart, centromereEnd).
        """
        gapPath = os.path.join(self.genomePath, self.gapFile)
        if os.path.isfile(gapPath):
            raise Exception('The gap file {0} already exists!'.format(gapPath))
        gapFile = open(os.path.join(self.genomePath, self.gapFile), 'w')
        for label, (i, j) in centromere_positions.iteritems():
            centromereStart = min(i, j)
            centromereEnd = max(i, j)
            gapFile.write(
                '0\t{0}\t{1}\t{2}\t0\tN\t0\tcentromere\tno\n'.format(
                    label, centromereStart, centromereEnd))

        gapFile.close()
        self._parseGapFile()

    def setResolution(self, resolution):
        """Set the resolution of genome binning and calculate the following 
        attributes:

        resolution : int
            The size of a bin for the binned values.

        chrmLensBin : array of int
            The lengths of chromosomes in bins.

        chrmStartsBinCont : array of int
            The positions of the first bins of the chromosomes in the
            concatenated genome.

        chrmEndsBinCont : array of int
            The positions of the last plus one bins of the chromosomes in the
            concatenated genome.

        chrmIdxBinCont : array of int
            The index of a chromosome in each bin of the concatenated genome.

        posBinCont : array of int
            The index of the first base pair in a bin in the concatenated
            genome.

        cntrMidsBinCont : array of int
            The position of the middle bin of a centromere in the concatenated
            genome.

        chrmArmLimitsBinCont: array of int
            The position of the chromosome arm borders in the concatenated
            genome.

        GCBin : list of arrays of float
            % of GC content of bins in individual chromosomes.

        unmappedBasesBin : list of arrays of int
            Number of bases with N's for each bin

        mappedBasesBin : list of arrays of int
            Number of sequenced bases for each bin

        binSizesbp : list of arrays of int
            Size of each bin. Is less than *resolution* for the last bin only.
        """

        if (resolution == -1) and hasattr(self, "resolution"):
            for i in ["chrmLensBin", "chrmStartsBinCont",
                      "chrmEndsBinCont", "numBins",
                      "chrmIdxBinCont", "posBinCont",
                      "cntrMidsBinCont", "chrmArmLimitsBinCont",
                      "GCBin",
                      "unmappedBasesBin", "binSizesBp",
                      "mappedBasesBin", "resolution"]:
                exec("del self.%s" % i)
            return

        self.resolution = int(resolution)

        # Bin chromosomes.
        self.chrmLensBin = self.chrmLens / self.resolution + 1
        self.chrmStartsBinCont = numpy.r_[0, numpy.cumsum(
            self.chrmLensBin)[:-1]]
        self.chrmEndsBinCont = numpy.cumsum(self.chrmLensBin)
        self.numBins = self.chrmEndsBinCont[-1]

        self.chrmIdxBinCont = numpy.zeros(self.numBins, int)
        for i in xrange(self.chrmCount):
            self.chrmIdxBinCont[
                self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]] = i

        self.posBinCont = numpy.zeros(self.numBins, int)
        for i in xrange(self.chrmCount):
            self.posBinCont[
                self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]] = (
                    self.resolution
                    * numpy.arange(-self.chrmStartsBinCont[i]
                                   + self.chrmEndsBinCont[i]))

        # Bin centromeres.
        self.cntrMidsBinCont = (self.chrmStartsBinCont
                                + self.cntrMids / self.resolution)
        self.chrmArmLimitsBinCont = numpy.zeros(self.chrmCount * 2 + 1, dtype=numpy.int)
        self.chrmArmLimitsBinCont[1::2] = self.cntrMidsBinCont
        self.chrmArmLimitsBinCont[2::2] = self.chrmEndsBinCont

        # Bin GC content.
        self.GCBin = self.getGCBin(self.resolution)
        self.unmappedBasesBin = self.getUnmappedBasesBin(self.resolution)
        self.binSizesBp = []
        for i in xrange(self.chrmCount):
            chromLen = self.chrmLens[i]
            cur = [self.resolution for _ in xrange(chromLen / self.resolution)]
            cur.append(chromLen % self.resolution)
            self.binSizesBp.append(numpy.array(cur))
        self.mappedBasesBin = [i[0] - i[1] for i in zip(
            self.binSizesBp, self.unmappedBasesBin)]

    def getUnmappedBases(self, chrmIdx, start, end):
        "Calculate the percentage of unmapped base pairs in a region."
        seq = self.seqs[chrmIdx][start:end]
        if len(seq.seq) == 0:
            return 0.0
        else:
            return (100.0 * (seq.seq.count('N') + seq.seq.count('n'))
                    / float(len(seq.seq)))

    def getGC(self, chrmIdx, start, end):
        """Calculate the GC content of the mapped part of a region. If there
        are no mapped base pairs, return 50%.
        """
        seq = self.seqs[chrmIdx][start:end]
        overall_GC = Bio.SeqUtils.GC(seq.seq)
        unmapped_content = self.getUnmappedBases(chrmIdx, start, end)

        if unmapped_content == 100.0:
            return -1.0
        else:
            corrected_GC = overall_GC * 100.0 / (100.0 - unmapped_content)
            return corrected_GC

    def clearCache(self):
        '''Delete the cached data in the genome folder.'''
        if hasattr(self, '_mymem'):
            self._mymem.clear()

    def setEnzyme(self, enzymeName):
        """Apply a specified restriction enzyme to the genomic sequences and 
        calculate the positions of restriction sites.

        The following attributes are set with this method:

        enzymeName : str
            The restriction enzyme used to find the restriction sites.

        rsites : list of arrays of int
            The indices of the first base pairs of restriction fragments
            in individual chromosomes.

        rfragMids : list of arrays of int
            The indices of the middle base pairs of restriction fragments
            in individual chromosomes.

        rsiteIds : array of int
            The position identifiers of the first base pairs of restriction
            fragments.

        rsiteMidIds : array of int
            The position identifiers of the middle base pairs of restriction
            fragments.

        rsiteChrms : array of int
            The indices of chromosomes for restriction sites in corresponding
            positions of rsiteIds and rsiteMidIds.
        """

        self.enzymeName = enzymeName

        self.rsites, self.rfragMids = self.getRsites(enzymeName)

        self.rsiteIds = numpy.concatenate(
            [self.rsites[chrm] + chrm * self.fragIDmult
             for chrm in xrange(self.chrmCount)])

        self.rfragMidIds = numpy.concatenate(
            [self.rfragMids[chrm] + chrm * self.fragIDmult
             for chrm in xrange(self.chrmCount)])

        self.rsiteChrms = numpy.concatenate(
            [numpy.ones(len(self.rsites[chrm]), int) * chrm
             for chrm in xrange(self.chrmCount)])

        assert (len(self.rsiteIds) == len(self.rfragMidIds))

    def hasEnzyme(self):
        return hasattr(self, "enzymeName")

    def splitByChrms(self, inArray):
        return [inArray[self.chrmStartsBinCont[i]:self.chrmEndsBinCont[i]]
                for i in xrange(self.chrmCount)]

    def upgradeMatrix(self, oldGenome):
        """Checks if old genome can be upgraded to new genome by truncation.
        If not, returns an array that can be used
        to upgrade chromosome positions.
        If upgrade not possible, raises an exception.

        Paramters
        ---------
        old Genome : Genome, or label2idx dictionary
            old genome from which upgrade is done

        Returns
        -------
        None : upgrade is possible by truncating chromosomes >= chromNum
        upgradeIndex : ndarray  upgrade is possible
        by newChrom = upgradeMatrix[oldChrom]

        Raises an exception when upgrade is not possible
        """

        if isinstance(oldGenome, Genome):
            oldGenome = oldGenome.idx2label
        if True in [i not in oldGenome.values() \
                    for i in self.idx2label.values()]:
            difference = [i for i in self.idx2label.values(
                ) if i not in oldGenome.values()]
            raise StandardError("Genome upgrade is not possible: " + \
                            repr(difference) + " are chromosomes"\
                            " that are missing in the old genome")
        if False not in [oldGenome[i] == self.idx2label[i] \
                         for i in self.idx2label.keys()]:
            return None
        oldLabelToIdx = dict([(oldGenome[i], i) for i in oldGenome.keys()])
        convertingArray = numpy.zeros(len(oldGenome.keys()), dtype=int) - 1
        for i in self.idx2label.values():
            convertingArray[oldLabelToIdx[i]] = self.label2idx[i]
        return convertingArray

    def checkReadConsistency(self, chromosomes, positions):
        """

        """
        chromSet = set(chromosomes)
        if 0 not in chromSet:
            warnings.warn("Chromosome zero not found! Are you using"\
                          " zero-based chromosomes?", UserWarning)
        if max(chromSet) >= self.chrmCount:
            raise StandardError("Chromosome number %d exceeds expected"
                                " chromosome count %d" % \
                                (max(chromSet), self.chrmCount))
        if max(chromSet) < self.chrmCount - 1:
            warnings.warn("More chromosomes in the genome (%d)  than we got"\
                          " (%d) ! Are you using proper genome?" % \
                          (self.chrmCount, max(chromSet) - 1))
        maxpositions = self.chrmLens[chromosomes]
        check = positions > maxpositions
        if check.any():  # found positions that exceeds chromosme length
            inds = numpy.nonzero(check)[0]
            inds = inds[::len(inds) / 10]
            for i in inds:
                raise StandardError("Position %d on chrm %d exceeds "\
                                    "maximum positions %d" % (
                        chromosomes[i], positions[i],
                        self.chrmLens[chromosomes[i]])
                                    )

    def getFragmentDistance(self, fragments1, fragments2, enzymeName):
        """returns distance between fragments
        measured in... fragments. (neighbors = 1, etc. )"""
        if not hasattr(self, "rfragMidIds"):
            self.setEnzyme(enzymeName)
        frag1ind = numpy.searchsorted(self.rfragMidIds, fragments1)
        frag2ind = numpy.searchsorted(self.rfragMidIds, fragments2)
        distance = numpy.abs(frag1ind - frag2ind)
        del frag1ind, frag2ind
        ch1 = fragments1 / self.fragIDmult
        ch2 = fragments2 / self.fragIDmult
        distance[ch1 != ch2] = 1000000
        return distance

    def getPairsLessThanDistance(self, fragments1, fragments2,
                                 cutoffDistance, enzymeName):
        """returns all possible pairs (fragment1,fragment2)
        with fragment distance less-or-equal than cutoff"""
        if not hasattr(self, "rfragMidIds"):
            self.setEnzyme(enzymeName)
        f1ID = numpy.searchsorted(self.rfragMidIds, fragments1)
        f2ID = numpy.searchsorted(self.rfragMidIds, fragments2)

        assert (fragments1[::100] - self.rfragMidIds[f1ID[::100]]).sum() == 0
        assert (fragments2[::100] - self.rfragMidIds[f2ID[::100]]).sum() == 0

        fragment2Candidates = numpy.concatenate(
            [f1ID + i for i in (range(-cutoffDistance, 0) + \
                                range(1, cutoffDistance + 1))])
        fragment1Candidates = numpy.concatenate(
            [f1ID for i in (range(-cutoffDistance, 0) + \
                            range(1, cutoffDistance + 1))])
        mask = numutils.arrayInArray(fragment2Candidates, f2ID)

        fragment2Real = fragment2Candidates[mask]
        fragment1Real = fragment1Candidates[mask]
        return  (self.rfragMidIds[fragment1Real],
                 self.rfragMidIds[fragment2Real])

    def _parseFixedStepWigAtKbResolution(self, filename, resolution):
        """Internal method for parsing fixedStep wig file
        and averaging it over every kb"""
        myfilename = filename
        if os.path.exists(filename) == False:
            raise StandardError("File not found!")
        M = self.maxChrmLen
        Mkb = int(M / resolution + 1)
        chromCount = self.chrmCount
        data = numpy.zeros(Mkb * self.chrmCount, float)
        resolution = int(resolution)
        if "X" in self.chrmLabels:
            useX = True
            Xnum = self.label2idx["X"] + 1  # wig uses zero-based counting
        else:
            useX = False
            Xnum = 0

        if "Y" in self.chrmLabels:
            useY = True
            Ynum = self.label2idx["Y"] + 1
        else:
            useY = False
            Ynum = 0

        if "M" in self.chrmLabels:
            useM = True
            Mnum = self.label2idx["M"] + 1
        else:
            useM = False
            Mnum = 0
        chromCount, useX, useY, useM, Ynum, Xnum, Mnum, myfilename
        code = r"""
        #line 14 "binary_search.py"
        using namespace std;
        int chrom=1;
        bool skip = false;
        int pos;
        int step;
        char line[50];
        char chromNum[10];
        const char * filename = myfilename.c_str();
        FILE *myfile;

        myfile = fopen(filename,"r");

        int breakflag = 0;

        while (fgets(line, 50, myfile) != NULL)
        {

          if (line[0] == 'f')
              {
              for (int j = 0;j<strlen(line);j++)
              {
              }
              if (breakflag == 1) break;
              sscanf(line,"fixedStep chrom=chr%s start=%d step=%d",
              &chromNum,&pos,&step);
              skip = false;
              chrom = atoi(chromNum);
              if (strcmp(chromNum ,"X") == 0)
               { chrom = Xnum; if (useX == false) skip = true;}
              if (strcmp(chromNum ,"Y") == 0)
               { chrom = Ynum; if (useY == false) skip = true;}
              if (strcmp(chromNum ,"M") == 0)
               { chrom = Mnum; if (useM == false) skip = true;}
              if ((chrom == 0) || (chrom > chromCount)) skip = true;
              if (skip == true) printf("Skipping chromosome %s\n", chromNum);

              cout << "working on chromosome  " << chrom << endl;
              continue;
              }
            if (skip == false)
            {
              double t;
              sscanf(line,"%lf",&t);
              data[Mkb * (chrom - 1) + pos / resolution] += t;
              pos+= step;
            }
        }
        """
        support = """
        #include <math.h>
        #include <iostream>
        #include <fstream>
        """
        weave.inline(code, ['myfilename', "data", "chromCount",
                            "useX", "useY", "useM",
                            "Xnum", "Ynum", "Mnum", "Mkb", "resolution"],
                     extra_compile_args=['-march=native -malign-double'],
                     support_code=support)

        datas = [data[i * Mkb:(i + 1) * Mkb] for i in xrange(self.chrmCount)]
        for chrom, track in enumerate(datas):
            if track[self.chrmLens[chrom] / resolution + 1:].sum() != 0:
                raise StandardError("Genome mismatch: entrees "\
                                    "in wig file after chromosome end!")
        datas = [numpy.array(i[:self.chrmLens[chrom] / resolution +
            1]) for chrom, i in enumerate(datas)]
        return datas

    def parseFixedStepWigAtKbResolution(self, filename, resolution=5000):
        "Returns averages of a fixedStepWigFile for all chromosomes"
        # At the first call the function rewrites itself with a memoized
        # private function.
        self.parseFixedStepWigAtKbResolution = self._memoize(
            '_parseFixedStepWigAtKbResolution')
        return self.parseFixedStepWigAtKbResolution(filename,
                                                    resolution=resolution)

    def _parseBigWigFile(self, filename, resolution=5000,
                         divideByValidCounts=False):
        import bx.bbi.bigwig_file
        from bx.bbi.bigwig_file import BigWigFile

        """
        Internal method for parsing bigWig files
        """
        data = []
        if type(filename) == str:
            bwFile = BigWigFile(open(filename))
        else:
            bwFile = BigWigFile(filename)
        print "parsingBigWigFile",
        assert isinstance(bwFile, bx.bbi.bigwig_file.BigWigFile)

        for i in xrange(self.chrmCount):
            chrId = "chr%s" % self.idx2label[i]
            print chrId,
            totalCount = int(numpy.ceil(self.chrmLens[i] / float(resolution)))
            values = numpy.zeros(totalCount, float)
            step = 500
            for i in xrange(totalCount / step):
                beg = step * i
                end = min(step * (i + 1), totalCount * resolution)
                summary = bwFile.summarize(chrId, beg *
                    resolution, end * resolution, end - beg)
                if summary is None:
                    continue
                stepValues = summary.sum_data
                stepCounts = summary.valid_count
                if divideByValidCounts == True:
                    stepValues = stepValues / stepCounts
                    stepCounts[stepCounts == 0] = 0
                values[beg:end] = stepValues
            if values.sum() == 0:
                raise  StandardError("Chromosome {0} is absent in bigWig"\
                                     " file!".format(chrId))
            data.append(values)

        return data

    def parseBigWigFile(self, filename, resolution=5000,
                        divideByValidCounts=False):
        """
        Parses bigWig file using bxPython build-in method "summary".
        Does it by averaging values over "resolution" long windows.

        If window has less than lowCountCutoff valid valies, it is discarded

        Parameters
        ----------

        filename : str or file object
            Incoming bigWig file
        lowCountCutoff : int, < resolution
            Ignore bins with less than cutoff valid bases
        resolution : int
            Find average signal over these bins
        divideByValidCounts : bool
            Divide  total coverage of the kb bin.

        Retruns
        -------
        List of numpy.arrays with average values for each chromosomes
        Length of each array is ceil(chromLens / resolution)
        """

        # At the first call the function rewrites itself with a memoized
        # private function.
        self.parseBigWigFile = self._memoize('_parseBigWigFile')
        return self.parseBigWigFile(filename,
                                    resolution, divideByValidCounts)
