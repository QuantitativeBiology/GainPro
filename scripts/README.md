# Scripts

This folder contains utility scripts for running and managing GainPro experiments.

## Available Scripts

### `multiple_runs.sh`

Runs the imputation model multiple times for statistical analysis or hyperparameter tuning.

**Usage:**
```bash
# Default: 50 runs with breast dataset parameters
./scripts/multiple_runs.sh

# Custom parameters file and number of runs
./scripts/multiple_runs.sh ./path/to/parameters.json 100
```

**Arguments:**
1. `parameters_file` (optional): Path to the JSON parameters file. Default: `./datasets/breast/parameters.json`
2. `num_runs` (optional): Number of runs to execute. Default: `50`
