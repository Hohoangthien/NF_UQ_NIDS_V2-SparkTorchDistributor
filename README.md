# GRU-Based Network Intrusion Detection System

A distributed machine learning pipeline for network intrusion detection using GRU (Gated Recurrent Unit) neural networks, built with Apache Spark and PyTorch.

## 🏗️ Architecture

The project is organized into modular components:

```
├── main.py                 # Main training script
├── config.py              # Configuration management
├── spark_utils.py         # Spark session utilities
├── hdfs_utils.py          # HDFS operations
├── model.py               # GRU model definition
├── data_processing.py     # Data loading and processing
├── training.py            # Training and evaluation logic
├── evaluation.py          # Model evaluation utilities
├── reporting.py           # Reporting and visualization
├── requirements.txt       # Python dependencies
├── run_training.sh        # Cluster training script
├── run_training_local.sh  # Local development script
└── README.md             # This file
```

## 🚀 Features

- **Distributed Training**: Uses Spark's TorchDistributor for scalable training
- **Optimized GRU Model**: Lightweight architecture designed for intrusion detection
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and classification reports
- **HDFS Integration**: Seamless data loading and result storage
- **Flexible Configuration**: Easy parameter tuning via command line arguments
- **Progress Tracking**: Real-time training progress and performance monitoring

## 📋 Requirements

### System Requirements
- Apache Spark 3.3+
- Hadoop with HDFS
- Python 3.8+
- YARN cluster (for distributed training)

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### Environment Variables
Set these in your environment or modify the run scripts:

```bash
# Data paths
export TRAIN_PATH="hdfs://master:9000/path/to/train_df.parquet"
export TEST_PATH="hdfs://master:9000/path/to/test_df.parquet"
export LABEL_INDEXER_PATH="hdfs://master:9000/path/to/label_indexer"
export OUTPUT_DIR="hdfs://master:9000/path/to/results"

# Training parameters
export NUM_PROCESSES=10
export EPOCHS=200
export BATCH_SIZE=32
export LEARNING_RATE=0.002
export HIDDEN_SIZE=64
export DROPOUT=0.2

# Python environment
export PYTHON_ENV="/path/to/your/python/env/bin/python"
```

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Number of training epochs |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.002 | AdamW optimizer learning rate |
| `hidden_size` | 64 | GRU hidden layer size |
| `dropout` | 0.2 | Dropout rate for regularization |

## 🏃‍♂️ Usage

### Cluster Training

1. **Make the script executable:**
   ```bash
   chmod +x run_training.sh
   ```

2. **Run distributed training:**
   ```bash
   ./run_training.sh
   ```

3. **Or with custom parameters:**
   ```bash
   NUM_PROCESSES=20 EPOCHS=300 BATCH_SIZE=64 ./run_training.sh
   ```

### Local Development

1. **Make the script executable:**
   ```bash
   chmod +x run_training_local.sh
   ```

2. **Update data paths in the script for local files**

3. **Run local training:**
   ```bash
   ./run_training_local.sh
   ```

### Manual Spark Submit

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 4G \
  --executor-cores 3 \
  --driver-memory 3G \
  main.py \
  --mode cluster \
  --num_processes 10 \
  --train_path "hdfs://master:9000/path/to/train.parquet" \
  --test_path "hdfs://master:9000/path/to/test.parquet" \
  --epochs 200 \
  --batch_size 32
```

## 📊 Data Format

The pipeline expects Spark DataFrame with the following schema:

```
├── scaled_features: VectorUDT (feature vector)
├── label: DoubleType (class label)
└── weight: DoubleType (sample weight)
```

Additional files needed:
- **Label Indexer**: Spark StringIndexerModel for class label mapping
- **Validation Data**: `val_df.parquet` in the same directory as training data

## 📈 Output

The pipeline generates comprehensive outputs:

### Model Files
- `best_model.pth`: Trained PyTorch model state dict
- `model_info.json`: Model architecture information

### Training Metrics
- `training_history.png`: Loss and accuracy curves
- `training_history.json`: Raw training metrics

### Evaluation Results
- `confusion_matrix.png`: Confusion matrix heatmap
- `confusion_matrix.json`: Confusion matrix data
- `classification_report.json`: Detailed per-class metrics
- `experiment_summary.json`: Overall performance summary

## 🎯 Model Architecture

```python
OptimizedGRUModel(
  (input_projection): Linear(in_features=input_size, out_features=hidden_size, bias=False)
  (gru): GRU(hidden_size, hidden_size, batch_first=True, bias=False)
  (dropout): Dropout(p=dropout)
  (classifier): Linear(in_features=hidden_size, out_features=num_classes, bias=False)
)
```

**Key Features:**
- Input projection layer for dimensionality adjustment
- Single-layer GRU for sequence modeling
- Dropout for regularization
- Bias-free layers for efficiency
- Gradient clipping for training stability

## 🔍 Monitoring

The pipeline provides real-time monitoring:

```
--- EPOCH 45/200 ---
  Val Loss: 0.1234 | Val Acc: 0.9567
  Train Time: 125.34s | Val Time: 12.45s
🎉 New best validation accuracy: 0.9567
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `batch_size`
   - Increase `executor-memory` and `memoryOverhead`
   - Reduce `num_processes`

2. **Data Loading Issues**
   - Verify HDFS paths are accessible
   - Check Spark SQL adaptive settings
   - Ensure proper permissions

3. **Distributed Training Failures**
   - Check network connectivity between nodes
   - Verify Python environment consistency
   - Increase timeout configurations

### Performance Tuning

1. **For Large Datasets:**
   ```bash
   export NUM_PROCESSES=20
   export BATCH_SIZE=16
   ```

2. **For Memory-Constrained Environments:**
   ```bash
   export BATCH_SIZE=8
   export HIDDEN_SIZE=32
   ```

3. **For Fast Convergence:**
   ```bash
   export LEARNING_RATE=0.001
   export DROPOUT=0.3
   ```

## 📚 Module Details

### `config.py`
- Command line argument parsing
- Configuration management
- Environment setup

### `spark_utils.py`
- Spark session initialization
- Metadata loading utilities

### `hdfs_utils.py`
- HDFS file operations
- Directory management
- Upload/download utilities

### `model.py`
- GRU model definition
- Model factory functions
- Architecture utilities

### `data_processing.py`
- Data partitioning for distributed training
- PyTorch dataset implementations
- DataLoader creation

### `training.py`
- Distributed training logic
- Epoch training loops
- Model evaluation

### `evaluation.py`
- Test set evaluation
- Model loading from HDFS
- Performance analysis

### `reporting.py`
- Visualization generation
- Report creation and saving
- Metrics tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Spark and PyTorch documentation
3. Open an issue on the repository
