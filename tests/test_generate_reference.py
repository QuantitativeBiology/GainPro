import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gainpro.dataset import Data
import numpy as np
import unittest
from gainpro.utils import create_csv
import torch
import pandas as pd
import random

class TestImputation(unittest.TestCase):
    def test_reference_generation(self):
        """Test the generation of the reference dataset process."""
        
        self.seed = 42  
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        dataset_df = pd.read_csv("breastMissing_20.csv")
        self.dataset = dataset_df.values

        hint_rate = 0.9
        miss_rate=0.1

        # Create an instance of Data
        data_instance = Data(self.dataset, miss_rate, hint_rate)

        # Call _create_ref() correctly on the instance
        data_instance._create_ref(miss_rate, hint_rate)

        # Get the reference dataset from the instance
        generate_reference = data_instance.ref_dataset
        
        #convert the reference dataset into a pandas dataframe
        df_reference = pd.DataFrame(generate_reference.numpy()) 


        missing_header = dataset_df.columns.tolist()
        create_csv(df_reference, "reference_generated", missing_header)
         
        output = pd.read_csv("output_generation_reference.csv")
        np.testing.assert_array_equal(df_reference, output.values, "Reference Dataset does not match expected output")



if __name__ == "__main__":
    unittest.main()