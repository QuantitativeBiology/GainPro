import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gainpro")))

from gainpro.dataset import Data
from gainpro.model import Network
from gainpro.hypers import Params
from gainpro.output import Metrics
import numpy as np
import unittest
import torch
import pandas as pd
import random
class TestImputation(unittest.TestCase):
    def setUp(self):
        """Set up reusable test data and parameters."""

        self.seed = 42 
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.dataset_path = "breastMissing_20.csv"
        self.ref_path = None
        self.imputed_file = "imputed"
        self.params = Params(
            input=self.dataset_path,
            output=self.imputed_file,
            ref=self.ref_path,
            output_folder=".",
            num_iterations=2001,
            batch_size=128,
            alpha=10,
            miss_rate=0.1,
            hint_rate=0.9,
            lr_D=0.001,
            lr_G=0.001,
            override=1,
            output_all=1,
        )

    def tearDown(self):
        """Clean up files created during testing."""
        if os.path.exists(self.imputed_file):
            os.remove(self.imputed_file)

    def test_imputation(self):
        """Test the imputation process."""
        # Load dataset and reference
        dataset_df = pd.read_csv(self.dataset_path)
        dataset = dataset_df.values  # Convert to numpy array

        # Extract headers 
        missing_header = dataset_df.columns.tolist()

        # Dummy network structures based on input dimensions
        input_dim = dataset.shape[1]  # Number of features
        h_dim = input_dim  # Hidden layer size
        net_G = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, input_dim),
            torch.nn.Sigmoid()
        )
        net_D = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, input_dim),
            torch.nn.Sigmoid()
        )

        # Initialize metrics
        metrics = Metrics(self.params)

        # Initialize the Network
        network = Network(hypers=self.params, net_G=net_G, net_D=net_D, metrics=metrics)
        self.assertIsNotNone(network, "Network initialization failed.")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Initialize Data
        data_obj = Data(
            dataset=dataset,
            miss_rate=0.2,
            hint_rate=0.9,
            ref = None  # Provide reference if available
        )
        self.assertIsNotNone(data_obj, "Data initialization failed.")

        # Perform training (imputation)
        try:
            network.evaluate(data = data_obj,missing_header = missing_header)
            network.train(data = data_obj, missing_header = missing_header)

            file1 = pd.read_csv(".imputed.csv")
            file2 = pd.read_csv("output_no_reference.csv")

            #np.testing.assert_array_equal(file1, file2, "Imputation performed successfully")
            #np.testing.assert_allclose(file1.values, file2.values, rtol=1e-5, atol=1e-8)
            
        except Exception as e:
            self.fail(f"Imputation failed with exception: {e}")

    
        "test the metrics class produced during imputation"
        self.assertEqual(metrics.loss_D.size, self.params.num_iterations)
        self.assertEqual(metrics.loss_G.size, self.params.num_iterations)
        self.assertEqual(metrics.ram.size, self.params.num_iterations)
        #self.assertEqual(metrics.loss_D[2000], 0.18696381151676178)


if __name__ == "__main__":
    unittest.main()
