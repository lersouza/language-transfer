#!/bin/bash

# First collect results from all the finetunes
python3 ./scripts/aggregate_results.py datasets
python3 ./scripts/aggregate_results.py finetune

# Preprocess the dataset
python3 ./scripts/preprocess.py ./results/finetune_data.csv \
    --output_file ./results/finetune_data_preprocessed.csv

# The calculate the data transfer metric
python3 ./scripts/calculate_data_transfer.py ./results/finetune_data_preprocessed.csv \
    --output_file ./results/finetune_data_transfer.csv

# Language distances 
python3 ./scripts/calculate_language_distances.py ./results/finetune_data_transfer.csv \
     --output_file ./results/finetune_data_transfer_distances.csv

# Build the final dataset
python3 ./scripts/build_tidy_dataset.py \
    ./results/finetune_data_transfer_distances.csv \
    ./results/datasets_data.csv \
    ./results/contamination/contamination/ \
    --output_file ./results/experiments_dataset.csv

