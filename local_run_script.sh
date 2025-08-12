#!/bin/bash

# Local GRU Model Training Script for Development and Testing
# Usage: ./run_training_local.sh

# Set default values for local testing
MODE="local"
NUM_PROCESSES=${NUM_PROCESSES:-2}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.002}
HIDDEN_SIZE=${HIDDEN_SIZE:-32}
DROPOUT=${DROPOUT:-0.2}

# Data paths - Update these according to your local setup
TRAIN_PATH=${TRAIN_PATH:-"file:///path/to/local/train_df.parquet"}
TEST_PATH=${TEST_PATH:-"file:///path/to/local/test_df.parquet"}
LABEL_INDEXER_PATH=${LABEL_INDEXER_PATH:-"file:///path/to/local/label_indexer"}
OUTPUT_DIR=${OUTPUT_DIR:-"file:///tmp/gru_results"}

# Python environment (use current environment for local)
PYTHON_ENV=${PYTHON_ENV:-$(which python)}

# Spark configuration for local mode
SPARK_SUBMIT_ARGS=(
    --master "local[*]"
    --name "GRU_NIDS_Local_Training_$(date +%Y%m%d_%H%M%S)"
    
    # Memory configuration (adjusted for local)
    --driver-memory 2G
    --conf spark.driver.maxResultSize=1G
    
    # Python environment
    --conf spark.pyspark.python=$PYTHON_ENV
    --conf spark.pyspark.driver.python=$PYTHON_ENV
    
    # Environment variables
    --conf spark.executorEnv.MPLCONFIGDIR=/tmp/matplotlib_cache
    --conf spark.executorEnv.OMP_NUM_THREADS=2
    
    # Performance optimization for local
    --conf spark.python.worker.reuse=true
    --conf spark.sql.adaptive.enabled=true
    --conf spark.sql.adaptive.coalescePartitions.enabled=true
    
    # Serialization
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
    
    # Partition configuration (smaller for local)
    --conf spark.sql.shuffle.partitions=4
    --conf spark.default.parallelism=4
)

# Application arguments
APP_ARGS=(
    --mode $MODE
    --num_processes $NUM_PROCESSES
    --train_path "$TRAIN_PATH"
    --test_path "$TEST_PATH"
    --label_indexer_path "$LABEL_INDEXER_PATH"
    --output_dir "$OUTPUT_DIR-$(date +%Y%m%d_%H%M%S)"
    --batch_size $BATCH_SIZE
    --epochs $EPOCHS
    --learning_rate $LEARNING_RATE
    --hidden_size $HIDDEN_SIZE
    --dropout $DROPOUT
)

# Print configuration
echo "=========================================="
echo "GRU Model Local Training Configuration"
echo "=========================================="
echo "Mode: $MODE"
echo "Processes: $NUM_PROCESSES"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Hidden Size: $HIDDEN_SIZE"
echo "Dropout: $DROPOUT"
echo "Train Path: $TRAIN_PATH"
echo "Test Path: $TEST_PATH"
echo "Output Dir: $OUTPUT_DIR-$(date +%Y%m%d_%H%M%S)"
echo "Python Env: $PYTHON_ENV"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p "$(dirname $OUTPUT_DIR)"

# Run the training
echo "Starting local training..."
spark-submit "${SPARK_SUBMIT_ARGS[@]}" main.py "${APP_ARGS[@]}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Local training completed successfully!"
else
    echo "❌ Local training failed!"
    exit 1
fi
