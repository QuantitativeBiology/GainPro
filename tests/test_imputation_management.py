import unittest
import numpy as np
import torch
import random
import pandas as pd

from gainpro.imputation_management import ImputationManagement 

class TestImputationManagement(unittest.TestCase):
    
    
    def test_correct_model(self):
        """test running a custom model after adding it"""

        self.seed = 42  
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Define a simple custom imputation function
        def custom_imputation(df):
            df_copy = df.copy()
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(0)
            return df_copy

        df_missing = pd.read_csv("breastMissing_20.csv")
        imputation_management = ImputationManagement("custom_model", df_missing, "breastMissing_20.csv")
        imputation_management.add_method("custom_model", custom_imputation)
        result = imputation_management.run_model("custom_model")
        self.assertIsNotNone(result, "Custom imputation should return a result")

    def test_incorrect_model(self):
        """test the class with an incorrect model"""
        df_missing = pd.read_csv("hela_missing_dann.csv")
        imputation_management = ImputationManagement("non_existing", df_missing, "hela_missing_dann.csv")
        with self.assertRaises(SystemExit):
            imputation_management.run_model("non_existing")
    
    def test_add_existing_model(self):
        """test adding an existing model"""
        df_missing = pd.read_csv("hela_missing_dann.csv")
        imputation_management = ImputationManagement("model_1", df_missing, "hela_missing_dann.csv")
        imputation_management.add_method("model_1", "model_1_function")
        self.assertEqual(imputation_management.dict_imputation_methods["model_1"], "model_1_function")

    def test_add_duplicate_model(self):
        """test to try adding model already known"""
        df_missing = pd.read_csv("breastMissing_20.csv")
        imputation_management = ImputationManagement("test_model", df_missing, "breastMissing_20.csv")
        imputation_management.add_method("test_model", lambda x: x)
        with self.assertRaises(SystemExit):
            imputation_management.add_method("test_model", "some_function")

if __name__ == '__main__':
    unittest.main()