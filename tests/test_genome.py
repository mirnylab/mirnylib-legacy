import os, sys
sys.path.insert(0, os.path.abspath('../'))

import unittest
import numpy 

import genome

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.db = genome.Genome('./data/', genomeName = None, gapfile = 'gap.txt',
                                chr_file_template = 'chr%s.fa')
        self.db.clear_cache()

    def test_init(self):
        self.assertEqual(
            self.db.id_to_idx,
            {'1': 0, '2': 1, '3': 2, 'X': 3})

        self.assertEqual(
            self.db.idx_to_id,
            {0: '1', 1: '2', 2: '3', 3: 'X'})

        self.assertTrue(numpy.all(numpy.equal(
            self.db.chromosomes, [49950, 49950, 24950, 49950])))

        self.assertTrue(numpy.all(numpy.equal(
            self.db.centromeres, [3920, 2815, 1500, 6012])))

        print self.db.getRsitesRfrags('HindIII')

if __name__ == '__main__':
        unittest.main()
