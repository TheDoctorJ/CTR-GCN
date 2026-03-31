import pickle
import unittest
from unittest.mock import patch

import numpy as np
from deepdiff import DeepDiff
from sympy import false

import data.ntu120.get_raw_skes_data as get_raw_skes_data
from data.ntu120.get_raw_skes_data_original import OriginalSkellyProcessor
from data.ntu120.get_raw_skes_data import SkeletonProcessor


class RawSkeletons(unittest.TestCase):
    def test_get_raw_skeletons_in_order(self):
        skelly = SkeletonProcessor()
        original_skelly = OriginalSkellyProcessor()
        skelly.setup_paths('./output/ntu120/modified', '../data/ntu120/')
        original_skelly.setup_paths('./output/ntu120/original', '../data/ntu120/')

        original_skelly.get_raw_skes_data(203)
        skelly.get_raw_skes_data(15, 203)

        original = dict()
        modified = dict()

        original['frames_cnt'] = np.loadtxt('./output/ntu120/original/raw_data/frames_cnt.txt', dtype=int)
        with open('./output/ntu120/original/raw_data/frames_drop_skes.pkl', 'rb') as f:
            original['frames_drop_skes_pkl'] = pickle.load(f)
        with open('./output/ntu120/original/raw_data/raw_skes_data.pkl', 'rb') as f:
            original['raw_skes_data_pkl'] = pickle.load(f)

        modified['frames_cnt'] = np.loadtxt('./output/ntu120/modified/raw_data/frames_cnt.txt', dtype=int)
        with open('./output/ntu120/modified/raw_data/frames_drop_skes.pkl', 'rb') as f:
            modified['frames_drop_skes_pkl'] = pickle.load(f)
        with open('./output/ntu120/modified/raw_data/raw_skes_data.pkl', 'rb') as f:
            modified['raw_skes_data_pkl'] = pickle.load(f)

        diff = DeepDiff(original, modified, ignore_order=False)
        self.assertEqual(diff, {})






if __name__ == '__main__':
    unittest.main()
