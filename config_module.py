"""
Configuration management module for GRU model training
"""
import argparse
import logging
import os


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRU Model Training Pipeline")
    
    # Mode and execution settings
    parser.add_argument(
        "--mode", 
        type=str, 
        default="local", 
        choices=["local", "cluster"],
        help="Execution mode: local or cluster"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=1,
        help="Number of processes for distributed training"
    )
    
    # Data paths
    parser.add_argument(
        "--train_path", 
        type=str, 
        required=True,
        help="Path to training data (parquet file)"
    )
    parser.add_argument(
        "--test_path", 
        type=str, 
        required=True,
        help="Path to test data (parquet file)"
    )
    parser.add_argument(
        "--label_indexer_path",
        type=str,
        default="hdfs://master:9000/usr/ubuntu/da_xu_ly/label_indexer/label_indexer",
        help="Path to label indexer model"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="hdfs://master:9000/usr/ubuntu/results",
        help="Output directory for results and models"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.002,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size for GRU model"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for regularization"
    )
    
    return parser.parse_args()


def setup_environment():
    """Setup environment variables"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Config:
    """Configuration class to hold all settings"""
    
    def __init__(self, args):
        # Execution settings
        self.mode = args.mode
        self.num_processes = args.num_processes
        
        # Data paths
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.label_indexer_path = args.label_indexer_path
        self.output_dir = args.output_dir
        
        # Training hyperparameters
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        
        # Model parameters (will be set later)
        self.num_features = None
        self.num_classes = None
        self.class_names = None
        
        # Temporary directories (will be set during execution)
        self.train_temp_dir = None
        self.val_temp_dir = None
        self.test_temp_dir = None
    
    def to_dict(self):
        """Convert config to dictionary for distributed training"""
        return {
            'mode': self.mode,
            'num_processes': self.num_processes,
            'train_path': self.train_path,
            'test_path': self.test_path,
            'label_indexer_path': self.label_indexer_path,
            'output_dir': self.output_dir,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_temp_dir': self.train_temp_dir,
            'val_temp_dir': self.val_temp_dir,
            'test_temp_dir': self.test_temp_dir,
        }
