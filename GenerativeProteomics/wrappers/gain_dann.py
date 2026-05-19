import os
import json
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download

class GainDannImputationModel():
    '''Imputation model using GAIN_DANN from Hugging Face.'''

    def run(self, df):
        """ Function to import and use the GAIN_DANN model from Hugging Face. """
    
        save_dir = "./GAIN_DANN_model"
        os.makedirs(save_dir, exist_ok=True)

        # Download files manually
        config_path = hf_hub_download(repo_id = "QuantitativeBiology/GAIN_DANN_model", filename="config.json", cache_dir = save_dir)
        weights_path = hf_hub_download(repo_id =f"QuantitativeBiology/GAIN_DANN_model", filename="pytorch_model.bin", cache_dir = save_dir)
        model_path = hf_hub_download(repo_id = f"QuantitativeBiology/GAIN_DANN_model", filename="modeling_gain_dann.py", cache_dir = save_dir)

        directory = os.path.dirname(model_path)

        # Add the directory containing 'modeling_gain_dann.py' to the Python path
        import sys
        sys.path.append(directory)  

        # Import the functions or classes inside the file
        from modeling_gain_dann import GainDANNConfig, GainDANN

        print("import successfully done")

        # Load config
        with open(config_path) as f:
            cfg = json.load(f)

        hela = df.iloc[:, 1:]
        input_dim = hela.shape[1]
        
        print(input_dim)
        cfg['input_dim'] = input_dim

        config = GainDANNConfig(**cfg)
        model = GainDANN(config)

        # Load raw state_dict
        state_dict = torch.load(weights_path, map_location="cpu")

        # Add "model." prefix to every key(because state_dict has the weights of GAIN_DANN but I added the Gain_DANN for the huggingFace accordance)
        renamed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        # Load with corrected keys
        model.load_state_dict(renamed_state_dict)

        model.eval()

        label_encoder = LabelEncoder()
        hela['Project'] = label_encoder.fit_transform(hela['Project'])

        x = torch.tensor(hela.values, dtype=torch.float32)

        with torch.no_grad():
            x_reconstructed, x_domain = model(x)

        print("x_reconstructed:", x_reconstructed)
        print("x_domain:", x_domain)

        pd.DataFrame(x_reconstructed.numpy()).to_csv("x_reconstructed.csv", index=False)
        pd.DataFrame(x_domain.numpy()).to_csv("x_domain.csv", index=False)