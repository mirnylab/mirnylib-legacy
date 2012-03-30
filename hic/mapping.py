'''
mapping - map raw Hi-C reads to a genome
========================================
'''
import os
import glob
import gzip
import re
import subprocess
import tempfile, atexit

import numpy as np

import Bio, Bio.SeqIO, Bio.Seq, Bio.Restriction
import pysam

from .. import h5dict
from .. import genome

##TODO: write some autodetection of chromosome lengthes base on genome folder
##TODO: throw an exception if no chromosomes found in chromosome folder

##TODO: fix #-to-ID correspondence for other species.

def _detect_quality_coding_scheme(in_fastq, num_entries = 10000):
    in_file = _gzopen(in_fastq)
    max_ord = 0
    min_ord = 256
    i = 0
    while True:
        line = in_file.readline()
        if not line or i > num_entries:
            break

        if not line.startswith('@'):
            raise Exception('%s does not comply with the FASTQ standards.')

        fastq_entry = [line, in_file.readline(), 
                       in_file.readline(), in_file.readline()]
        min_ord = min(min_ord, min(ord(j) for j in fastq_entry[3].strip()))
        max_ord = max(max_ord, max(ord(j) for j in fastq_entry[3].strip()))

        i += 1

    return min_ord, max_ord

def _gzopen(path):
    if path.endswith('.gz'):
        return gzip.open(path)
    else:
        return open(path)

def _line_count(path):
    '''Count the number of lines in a file. The function was posted by
    Mikola Kharechko on Stackoverflow.
    '''

    f = _gzopen(path)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines

def _slice_file(in_path, out_path, first_line, last_line):
    '''Slice lines from a large file. 
    The line numbering is as in Python slicing notation.
    '''
    f = _gzopen(in_path)                  
    output = _gzopen(out_path, 'w')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization
    write_f = output.write # loop optimization
    scanning = True

    buf = read_f(buf_size)
    while buf:
        eols_in_buf = buf.count('\n')

        if scanning:
            if lines + eols_in_buf > first_line:
               write_f('\n'.join(buf.split('\n')[
                   first_line - lines : min(last_line - lines, eols_in_buf)]))
               scanning = False

        else:
           if lines + eols_in_buf < last_line: 
               write_f(buf)
           else:
               write_f('\n'.join(buf.split('\n')[:last_line - lines]))
               break

        lines += eols_in_buf
        buf = read_f(buf_size)

    output.write('\n')
    output.close()

def filter_fastq(ids, in_fastq, out_fastq):
    '''Filter FASTQ sequences by their IDs.

    Read entries from **in_fastq** and store in **out_fastq** only those
    the whose ID are in **ids**.
    '''
    out_file = _gzopen(out_fastq, 'w')
    in_file = _gzopen(in_fastq)
    while True:
        line = in_file.readline()
        if not line:
            break

        if not line.startswith('@'):
            raise Exception('%s does not comply with the FASTQ standards.')

        fastq_entry = [line, in_file.readline(), 
                       in_file.readline(), in_file.readline()]
        read_id = line.split()[0][1:]
        if read_id.endswith('/1') or read_id.endswith('/2'):
            read_id = read_id[:-2]
        if read_id in ids:
            out_file.writelines(fastq_entry)

def filter_unmapped_fastq(in_fastq, in_sam, nonunique_fastq):
    '''Read raw sequences from **in_fastq** and alignments from 
    **in_sam** and save the non-uniquely aligned and unmapped sequences
    to **unique_sam**.
    '''
    samfile = pysam.Samfile(in_sam)

    nonunique_ids = set()
    for read in samfile:
        tags_dict = dict(read.tags)
        read_id = read.qname
        # If exists, the option 'XS' contains the score of the second 
        # best alignment. Therefore, its presence means non-unique alignment.
        if 'XS' in tags_dict or read.is_unmapped:
            nonunique_ids.add(read_id)

    filter_fastq(nonunique_ids, in_fastq, nonunique_fastq)

def iterative_mapping(bowtie_path, genome_path, fastq_path, out_sam_path,
                      min_seq_len, len_step, **kwargs):
    '''Map raw HiC reads iteratively with bowtie2.
    http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml

    1. Truncate the sequences to the first N = **min_seq_len** base pairs,
       starting at the **seq_start** position.
    2. Map the sequences using bowtie2.
    3. Store the uniquely mapped sequences in a SAM file at **out_sam_path**.
    4. Go to the step 1, increase the truncation length N by **len_step** base 
       pairs, and map the non-mapped and non-uniquely mapped sequences,
       ...
       Stop when the 3' end of the truncated sequence reaches the **seq_end**
       position.
       
    Parameters
    ----------

    bowtie_path : str
        The path to the bowtie2 executable.

    genome_path : str
        The path to the bowtie2 genome index. Since the index consists of 
        several files with the different suffices (e.g., hg18.1.bt2, 
        hg18.2.bt.2), provide only the common part (hg18).
        
    fastq_path : str
        The path to the input FASTQ file.

    out_sam_path : str
        The path to the output SAM file.

    min_seq_len : int
        The truncation length at the first iteration of mapping.

    len_step : int
        The increase in truncation length at each iteration.

    seq_start, seq_end : int, optional
        Slice the FASTQ sequences at [seq_start:seq_end]. Default is [O:None].

    nthreads : int, optional
        The number of Bowtie2 threads. Default is 8

    bowtie_flags : str, optional
        Extra command-line flags for Bowtie2.

    max_reads_per_chunk : int, optional
        If positive then split input into several chunks with 
        `max_reads_per_chunk` each and map them separately. Use for large 
        datasets and low-memory machines.

    '''

    seq_start = kwargs.get('seq_start', 0)
    seq_end = kwargs.get('seq_end', None)
    nthreads = kwargs.get('nthreads', 4)
    max_reads_per_chunk = kwargs.get('max_reads_per_chunk', -1)
    bowtie_flags = kwargs.get('bowtie_flags', '')

    # Remove the temporary directory at exit from the top function.
    if max_reads_per_chunk > 0:
        num_lines = _line_count(fastq_path)
        kwargs['max_reads_per_chunk'] = -1
        for i in range(num_lines / 4 / max_reads_per_chunk + 1):
            fastq_chunk_path = fastq_path + '.%d' % i
            _slice_file(fastq_path, fastq_chunk_path, 4 * i * max_reads_per_chunk, 
                        4 * (i + 1) * max_reads_per_chunk)
            iterative_mapping(bowtie_path, genome_path, fastq_chunk_path, 
                              out_sam_path + '.%d' % i, min_seq_len, len_step,
                              **kwargs)
        return 

    raw_seq_len = len(Bio.SeqIO.parse(_gzopen(fastq_path), 'fastq').next().seq)
    if (seq_start < 0 
        or seq_start > raw_seq_len 
        or (seq_end and seq_end > raw_seq_len)):
        raise Exception('An incorrect trimming region is supplied: [%d, %d), '
                        'the raw sequence length is %d' % (
                            seq_start, seq_end, raw_seq_len))
    local_seq_end = min(raw_seq_len, seq_end) if seq_end else raw_seq_len

    if min_seq_len <= local_seq_end - seq_start: 
        trim_5 = seq_start
        #trim_3 = raw_seq_len - local_seq_end
        trim_3 = raw_seq_len - seq_start - min_seq_len
        bowtie_command = (
            ('time %s -x %s --very-sensitive '#--score-min L,-0.6,-0.2 '
             '-q %s -5 %s -3 %s -p %s %s > %s') % (
                bowtie_path, genome_path, fastq_path, 
                str(trim_5), str(trim_3), str(nthreads), bowtie_flags,
                out_sam_path))

        print 'Map reads:', bowtie_command
        subprocess.call(bowtie_command, shell=True)

        print ('Save unique aligments and send the '
               'non-unique ones to the next iteration')

        #unmapped_fastq_path = fastq_path # Testing
        unmapped_fastq_path = os.path.join(
            tempfile.gettempdir(), fastq_path + '.%d' % min_seq_len)
        filter_unmapped_fastq(fastq_path, out_sam_path, unmapped_fastq_path)
        atexit.register(lambda: os.remove(unmapped_fastq_path))

        iterative_mapping(bowtie_path, genome_path, unmapped_fastq_path, 
                          out_sam_path + '.%d' % min_seq_len,
                          min_seq_len = min_seq_len + len_step, 
                          len_step=len_step, **kwargs)
     
def fill_rsites(lib, db_dir_path, enzyme_name, min_frag_size = None):
    '''Private: assign mapped reads to restriction fragments by 
    their 5' end position.

    Parameters
    ----------

    lib : dict
        A library of mapped Hi-C molecules. Modified by the function.

    db_dir_path
        
    '''
    if enzyme_name not in Bio.Restriction.AllEnzymes:
        raise Exception('Enzyme is not found in the library: %s' % (enzyme_name,))

    genomeDb = genome.Genome(db_dir_path)
    genomeDb.setEnzyme(enzyme_name)

    rsite_size = eval('len(Bio.Restriction.%s.site)' % enzyme_name)
    if min_frag_size is None:
        _min_frag_size = rsite_size / 2.0
    else:
        _min_frag_size = min_frag_size
        
    _find_rfrags_inplace(lib, genomeDb, _min_frag_size, 1)
    _find_rfrags_inplace(lib, genomeDb, _min_frag_size, 2)

    return lib

def _find_rfrags_inplace(lib, genome, min_frag_size, side):
    '''Private: assign mapped reads to restriction fragments by 
    their 5' end position.
    '''
    side = str(side) 

    chrms = lib['chrms' + side]
    rfragIdxs = np.zeros(len(chrms), dtype=np.int64)
    rsites = np.zeros(len(chrms), dtype=np.int64)
    uprsites = np.zeros(len(chrms), dtype=np.int64)
    downrsites = np.zeros(len(chrms), dtype=np.int64)

    # If the fragment was not mapped.
    rfragIdxs[chrms == -1] = -1
    rsites[chrms == -1] = -1
    uprsites[chrms == -1] = -1
    downrsites[chrms == -1] = -1

    cuts = lib['cuts' + side]
    strands = lib['strands' + side]
    for chrm_idx in xrange(genome.chrmCount):
        all_rsites = np.r_[0, genome.rsites[chrm_idx]]
        idxs = (chrms == chrm_idx)

        # Find the indexes of the restriction fragment...
        rfragIdxs[idxs] = np.searchsorted(all_rsites, cuts[idxs]) - 1
        uprsites[idxs] = all_rsites[rfragIdxs[idxs]]
        downrsites[idxs] = all_rsites[rfragIdxs[idxs] + 1] 
        rsites[idxs] = np.where(strands[idxs], downrsites[idxs], uprsites[idxs])

        too_close = (np.abs(rsites[idxs] - cuts[idxs]) <= min_frag_size)
        too_close_idxs = np.where(idxs)[0][too_close]
        rfragIdxs[too_close_idxs] += strands[too_close_idxs] * 2 - 1
        uprsites[too_close_idxs] = all_rsites[rfragIdxs[too_close_idxs]]
        downrsites[too_close_idxs] = all_rsites[rfragIdxs[too_close_idxs] + 1]
        rsites[too_close_idxs] = np.where(
            strands[too_close_idxs],
            downrsites[too_close_idxs], 
            uprsites[too_close_idxs])

    lib['rfragIdxs' + side] = rfragIdxs
    lib['uprsites' + side] = uprsites
    lib['downrsites' + side] = downrsites
    lib['rsites' + side] = rsites

def _parse_ss_sams(sam_wildcard, out_dict, 
                   max_seq_len = -1, reverse_complement=False):
    """Parse SAM files with single-sided reads.
    """
    def _for_each_unique_read(sam_wildcard, action):
        for sam_path in glob.glob(sam_wildcard):
            for read in pysam.Samfile(sam_path):
                # Skip non-mapped reads...
                if read.is_unmapped:
                    continue
                # ...non-uniquely aligned...
                for tag in read.tags:
                    if tag[0] == 'XS':
                        continue
                # ...or those not belonging to the target chromosome. 
                action(read) 

    print('Counting stats...')
    # Calculate reads statistics.
    def _count_stats(read):
        # In Python, function is an object and can have an attribute.
        # We are using the .cache attribute to store the stats.
        _count_stats.id_len = max(_count_stats.id_len,
                                  len(read.qname))
        _count_stats.seq_len = max(_count_stats.seq_len,
                                   len(read.seq))
        _count_stats.num_reads += 1
    _count_stats.id_len = 0
    _count_stats.seq_len = 0
    _count_stats.num_reads = 0
    _for_each_unique_read(sam_wildcard, action=_count_stats)
    sam_stats = {'id_len': _count_stats.id_len,
                 'seq_len':_count_stats.seq_len,
                 'num_reads':_count_stats.num_reads}
    if max_seq_len > 0:
        sam_stats['seq_len'] = min(max_seq_len, sam_stats['seq_len'])
    print('Done!')

    # Read and save each type of data separately.
    def _write_to_array(read, array, value):
        array[_write_to_array.i] = value
        _write_to_array.i += 1
    
    # ...chromosome ids
    buf = np.zeros((sam_stats['num_reads'],), dtype=np.int8)
    _write_to_array.i = 0
    _for_each_unique_read(sam_wildcard,
        action=lambda read: _write_to_array(read, buf, read.tid))
    out_dict['chrms'] = buf

    # ...strands
    buf = np.zeros((sam_stats['num_reads'],), dtype=np.bool)
    _write_to_array.i = 0
    _for_each_unique_read(sam_wildcard,
        action=lambda read: _write_to_array(read, buf, not read.is_reverse))
    out_dict['strands'] = buf

    # ...cut sites
    buf = np.zeros((sam_stats['num_reads'],), dtype=np.int64)
    _write_to_array.i = 0
    _for_each_unique_read(sam_wildcard,
        action=
            lambda read: _write_to_array(read, buf, read.pos + (len(read.seq) if read.is_reverse else 0)))
    out_dict['cuts'] = buf
    
    # ...sequences
    buf = np.zeros((sam_stats['num_reads'],), dtype='|S%d' % sam_stats['seq_len'])
    _write_to_array.i = 0
    _for_each_unique_read(sam_wildcard,
        action=
            lambda read: _write_to_array(read, buf, Bio.Seq.reverse_complement(read.seq) if read.is_reverse and reverse_complement else read.seq))
    out_dict['seqs'] = buf

    # and ids.
    buf = np.zeros((sam_stats['num_reads'],), dtype='|S%d' % sam_stats['id_len'])
    _write_to_array.i = 0
    _for_each_unique_read(sam_wildcard,
        action=lambda read: _write_to_array(read, buf, read.qname))
    out_dict['ids'] = buf

    return out_dict

def parse_sam(sam_wildcard1, sam_wildcard2, out_dict,
              max_seq_len = -1, reverse_complement=False, keep_ids=False):
    # Parse the single-sided reads.
    ss_lib = {}
    ss_lib[1] = h5dict.h5dict()
    ss_lib[2] = h5dict.h5dict()
    _parse_ss_sams(sam_wildcard1, ss_lib[1], 
                   1 if not max_seq_len else max_seq_len, reverse_complement)
    _parse_ss_sams(sam_wildcard2, ss_lib[2],
                   1 if not max_seq_len else max_seq_len, reverse_complement)

    # Determine the number of double-sided reads.
    all_ids = np.unique(np.concatenate((ss_lib[1]['ids'], ss_lib[2]['ids'])))
    tot_num_reads = all_ids.shape[0]

    # Pair single-sided reads and write into the output.
    for i in [1,2]:
        sorting = np.searchsorted(all_ids, ss_lib[i]['ids'])
        for key in ss_lib[i].keys():
            # Don't save ids and seqs if not requested.
            if key=='ids' and not keep_ids:
                continue
            if key=='seq' and not max_seq_len:
                continue

            # The default value is -1 for an undefined cut site and chromosome
            # and 0 for other data.
            if key=='cuts' or key=='chrms':
                buf = -1 * np.ones(shape=tot_num_reads, 
                                   dtype=ss_lib[i].value_dtype(key))
            else:
                buf = np.zeros(shape=tot_num_reads, 
                               dtype=ss_lib[i].value_dtype(key))

            buf[sorting] = ss_lib[i][key]
            out_dict[key + str(i)] = buf
            del buf

    return out_dict

