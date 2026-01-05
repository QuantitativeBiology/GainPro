import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from gainpro.dataset import Data, generate_hint
import numpy as np
import unittest
from gainpro.utils import create_csv
import torch
import pandas as pd
import random

class TestImputation(unittest.TestCase):
    def test_hint_generation(self):
        """Test the hint generation process."""

        self.seed = 42  
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        dataset_df = pd.read_csv("breastMissing_20.csv")
        dataset = dataset_df.values

        mask = np.where(np.isnan(dataset), 0.0, 1.0)
        hint_rate = 0.9

        file1 = generate_hint(mask, hint_rate)
        missing_header = dataset_df.columns.tolist()
        create_csv(file1, "hint_matrix", missing_header)
    
        output = pd.read_csv("output_hint.csv")
        np.testing.assert_array_equal(file1, output.values, "Hint matrix does not match expected output")


if __name__ == "__main__":
    unittest.main()


