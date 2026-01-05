#!/bin/bash
# Script to run multiple imputation training runs
# Usage: ./multiple_runs.sh [parameters_file] [num_runs]

set -euo pipefail  # Exit on error, undefined vars, pipe failures

PARAMS_FILE=${1:-"./datasets/breast/parameters.json"}
NUM_RUNS_INPUT=${2:-50}

# Validate NUM_RUNS is a positive integer
if ! [[ "$NUM_RUNS_INPUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_runs must be a positive integer (got: $NUM_RUNS_INPUT)" >&2
    exit 1
fi

# Validate PARAMS_FILE exists
if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "Error: Parameters file not found: $PARAMS_FILE" >&2
    exit 1
fi

NUM_RUNS=$NUM_RUNS_INPUT

echo "Running $NUM_RUNS imputation runs with parameters: $PARAMS_FILE"

# Use C-style for loop to avoid command injection (safer than $(seq ...))
for ((run=1; run<=NUM_RUNS; run++))
do
    echo "Running run = $run / $NUM_RUNS"
    gainpro gain --parameters "$PARAMS_FILE"
done

echo "Completed all $NUM_RUNS runs"
