from pathlib import Path
import yaml

import pipelines.dataset_pipeline as pipeline

def test_incorrect_filepath():
    "Expected to raise FileNotFoundError"

    print("")

    file_path = Path("./toy_hela.h5ad")
    print("File: ", file_path)

    try:
        pipeline.run_dataset_pipeline(file_path)
    except FileNotFoundError:
        print("Raised FileNotFoundError exception!")

# def test_load_anndata():
#     print("")

#     file_path = Path("../data/raw/HeLa/HeLa_datasets_combined.h5ad")
#     print("File: ", file_path)
#     print("Dataset ", file_path.stem)

#     df = pipeline.load_anndata(file_path)

#     assert df.shape == (4820, 11013)

def test_load_csv():
    print("")

    file_path = Path("./small_toy_hela.csv")
    print("File: ", file_path)
    print("Dataset ", file_path.stem)

    df = pipeline.load_csv(file_path)

    assert df.shape == (25, 10)

def read_config(config_path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    #todo check format of the config
    return config

def test_save_datasets():
    print("")

    config_file = "../configs/prepare_data.yaml"

    pipeline.run_dataset_pipeline(config_file)

    config = read_config(config_file)
    missingness_levels = config["missingness levels"]

    for level in missingness_levels:
        rate = int(level * 100)
        assert Path("../data/processed/small_toy_hela").is_dir()
        assert Path(f"../data/processed/small_toy_hela/miss_{rate}/reference.csv").is_file()
        assert Path(f"../data/processed/small_toy_hela/miss_{rate}/missing.csv").is_file()
        assert Path(f"../data/processed/small_toy_hela/miss_{rate}/mask.csv").is_file()
        assert Path(f"../data/processed/small_toy_hela/miss_{rate}/domain.csv").is_file()
