#!/bin/bash
# Model pipeline script for training and evaluating Temporal GNN

echo "Starting Temporal GNN model pipeline..."

set -e

# Step 1: Data Preprocessing
echo "Preprocessing data..."
python3 src/data_prep.py

# Step 2: Model Training
echo "Training Temporal GNN model..."
python3 src/train.py

# Step 3: Model Evaluation
echo "Evaluating model..."
python3 src/evaluate.py

echo "Temporal GNN model pipeline completed."