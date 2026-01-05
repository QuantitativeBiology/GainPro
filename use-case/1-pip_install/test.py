from gainpro import utils, Network, Params, Metrics, Data
import torch
import argparse
import os


def test_network():
    """ Test to showcase how to import and use the classes and functions of the gainpro package. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, help = "indicates the model to use")
    return parser.parse_args()

if __name__ == "__main__":
    # Load the dataset
    args = test_network()
    dataset_path = "PXD004452-8c3d7d43-b1e7-4a36-a430-23e41bcbe07c.absolute.tsv" 
    if args.model is None :
        ref_path = None  # Reference complete dataset
        
        # Load dataset 
        dataset_df = utils.build_protein_matrix(dataset_path)
        dataset = dataset_df.values  # Convert to numpy array

        
        # Extract headers (missing_header)
        missing_header = dataset_df.columns.tolist()

        # Define parameters for testing
        params = Params(
            input=dataset_path,
            output="imputed.csv",
            ref=ref_path,
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
        metrics = Metrics(params)

        # Initialize the Network
        network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

        # Initialize Data
        data = Data(
            dataset=dataset,
            miss_rate=0.2,
            hint_rate=0.9,
            ref = None  
        )
        
        # Perform training (imputation)
        print("Running imputation...")
        try:
            network.evaluate(data=data, missing_header=missing_header)  
            network.train(data=data, missing_header=missing_header) 
            print("Imputation completed successfully!")
        except Exception as e:
            print(f"Error during imputation: {e}")








