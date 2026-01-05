import unittest
import json
import os
from tempfile import NamedTemporaryFile
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gainpro.hypers import Params 

class TestParams(unittest.TestCase):
    
    def setUp(self):
        """Set up a temporary JSON file for testing"""
        self.test_params = {
            "input": "data.csv",
            "output": "imputed_data",
            "ref": "ref_data.csv",
            "output_folder": "./results/",
            "num_iterations": 1500,
            "batch_size": 64,
            "alpha": 5,
            "miss_rate": 0.2,
            "hint_rate": 0.8,
            "lr_D": 0.0005,
            "lr_G": 0.0005,
            "override": 1,
            "output_all": 1
        }
        
        # Create a temporary JSON file
        self.temp_file = NamedTemporaryFile(delete=False, suffix=".json")
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.test_params, f)
        self.temp_file.close()

    def test_read_json(self):
        """Test reading JSON file"""
        params = Params._read_json(self.temp_file.name)
        self.assertEqual(params, self.test_params)

    def test_initialization(self):
        """Test default initialization"""
        params = Params()
        self.assertEqual(params.input, None)
        self.assertEqual(params.output, "imputed")
        self.assertEqual(params.ref, None)
        self.assertEqual(params.output_folder, os.getcwd() + "/results/")
        self.assertEqual(params.batch_size, 128)
        self.assertEqual(params.lr_D, 0.001)
        self.assertEqual(params.lr_G, 0.001)
        self.assertEqual(params.alpha, 10)
        self.assertEqual(params.miss_rate, 0.1)
        self.assertEqual(params.hint_rate, 0.9)
        self.assertEqual(params.override, 0)
        self.assertEqual(params.output_all, 0)
        
    def test_read_hyperparameters(self):
        """Test reading hyperparameters from JSON file"""
        params = Params.read_hyperparameters(self.temp_file.name)
        self.assertEqual(params.input, "data.csv")
        self.assertEqual(params.output, "imputed_data")
        self.assertEqual(params.ref, "ref_data.csv")
        self.assertEqual(params.output_folder, "./results/")
        self.assertEqual(params.num_iterations, 1500)
        self.assertEqual(params.batch_size, 64)
        self.assertEqual(params.alpha, 5)
        self.assertEqual(params.miss_rate, 0.2)
        self.assertEqual(params.hint_rate, 0.8)
        self.assertEqual(params.lr_G, 0.0005)
        self.assertEqual(params.lr_D, 0.0005)
        self.assertEqual(params.override, 1)
        self.assertEqual(params.output_all, 1)

    def test_update_hypers(self):
        """Test updating hyperparameters dynamically"""
        params = Params()
        params.update_hypers(batch_size=256, lr_G=0.002, non_existing_param=10)
        
        self.assertEqual(params.batch_size, 256)
        self.assertEqual(params.lr_G, 0.002)
        with self.assertRaises(AttributeError):
            getattr(params, "non_existing_param")  

    def tearDown(self):
        """Clean up the temporary file after test execution"""
        os.remove(self.temp_file.name)
        
    def test_invalid_json(self):
        """Test handling of invalid JSON file"""
        with self.assertRaises(FileNotFoundError):
            Params.read_hyperparameters("non_existent_file.json")

if __name__ == '__main__':
    unittest.main()