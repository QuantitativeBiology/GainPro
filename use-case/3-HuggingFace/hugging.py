from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download
import torch
import os
import json
import pandas as pd
import argparse


def test_network():
    """ Test to showcase how to import and use the classes and functions of the gainpro package. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, help = "indicates the model to use")
    return parser.parse_args()

if __name__ == "__main__":

    args = test_network()
    if args.model is not None:

        save_dir = "./GAIN_DANN_model"
        os.makedirs(save_dir, exist_ok=True)

        # Download files manually
        config_path = hf_hub_download(repo_id = f"QuantitativeBiology/{args.model}", filename="config.json", cache_dir = save_dir)
        weights_path = hf_hub_download(repo_id =f"QuantitativeBiology/{args.model}", filename="pytorch_model.bin", cache_dir = save_dir)
        model_path = hf_hub_download(repo_id = f"QuantitativeBiology/{args.model}", filename="modeling_gain_dann.py", cache_dir = save_dir)

        print("config_path:", config_path)
        print("weights_path:", weights_path)
        print("model_path", model_path)

        directory = os.path.dirname(model_path)
        print("directory:", directory)

        import sys
        sys.path.append(directory)  # Add the directory containing 'modeling_gain_dann.py' to the Python path

        if args.model == "GAIN_DANN_model":

            from modeling_gain_dann import GainDANNConfig, GainDANN


            print("import successfully done")

            # Load config
            with open(config_path) as f:
                cfg = json.load(f)

            hela = pd.read_csv("hela_missing_dann.csv", index_col=0)

            input_dim = hela.shape[1]
            
            print(input_dim)
            cfg['input_dim'] = input_dim

            config = GainDANNConfig(**cfg)
            model = GainDANN(config)

            # Load raw state_dict
            state_dict = torch.load(weights_path, map_location="cpu")

            # Add "model." prefix to every key (because state_dict has the weights of GAIN_DANN but I added the Gain_DANN for the huggingFace accordance)
            renamed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

            # Load with corrected keys
            model.load_state_dict(renamed_state_dict)

            model.eval()

            print(hela.dtypes)

            label_encoder = LabelEncoder()
            hela['Project'] = label_encoder.fit_transform(hela['Project'])

            x = torch.tensor(hela.values, dtype=torch.float32)

            with torch.no_grad():
                x_reconstructed, x_domain = model(x)

            pd.DataFrame(x_reconstructed.numpy()).to_csv("x_reconstructed.csv", index=False)
            pd.DataFrame(x_domain.numpy()).to_csv("x_domain.csv", index=False)
