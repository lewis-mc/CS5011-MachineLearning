#!/bin/bash

# Paths to datasets
TRAIN_VALUES="train_values_modified.csv"
TRAIN_LABELS="train_labels.csv"
TEST_VALUES="test_values.csv"

# Generate output filename based on the combination used
OUTPUT_FILE="../output/${scaler}${encoder}${model}.csv"

# Run the Python script
echo "Running: $scaler, $encoder, $model"
python updated_optuna.py "$TRAIN_VALUES" "$TRAIN_LABELS" "$TEST_VALUES" "$OUTPUT_FILE" --hpo

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo "Error encountered with: $scaler, $encoder, $model"
fi

echo "All experiments completed!"