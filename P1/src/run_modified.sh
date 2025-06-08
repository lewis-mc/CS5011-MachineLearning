#!/bin/bash

# Define all possible values for preprocessing and models
SCALERS=("None" "StandardScaler")

ENCODERS=("OneHotEncoder" "OrdinalEncoder" "TargetEncoder")

MODELS=("LogisticRegression" "RandomForestClassifier" "GradientBoostingClassifier" "HistGradientBoostingClassifier" "MLPClassifier")


# Paths to datasets
TRAIN_VALUES="train_values_modified.csv"
TRAIN_LABELS="train_labels.csv"
TEST_VALUES="test_values.csv"

# Loop through all combinations of scalers, encoders, and models
for scaler in "${SCALERS[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
        for model in "${MODELS[@]}"; do
            # Generate output filename based on the combination used
            OUTPUT_FILE="../output_modified/modified_${scaler}${encoder}${model}.csv"
            
            # Run the Python script
            echo "Running: $scaler, $encoder, $model"
            python train-and-predict.py "$TRAIN_VALUES" "$TRAIN_LABELS" "$TEST_VALUES" "$scaler" "$encoder" "$model" "$OUTPUT_FILE"
            
            # Check if the script executed successfully
            if [ $? -ne 0 ]; then
                echo "Error encountered with: $scaler, $encoder, $model"
            fi
        done
    done
done

echo "All experiments completed!"