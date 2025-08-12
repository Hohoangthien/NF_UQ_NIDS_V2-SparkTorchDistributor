#!/usr/bin/env python3
"""
Main training script for GRU-based intrusion detection model
Usage: spark-submit [spark-options] main.py [script-arguments]
"""

import os
import sys
from datetime import datetime

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import setup_logging, setup_environment, parse_args, Config
from spark_utils import init_spark, load_metadata
from data_processing import prepare_data_partitions
from training import training_function
from evaluation import evaluate_model_on_test_set
from reporting import save_model_performance_summary
from hdfs_utils import cleanup_hdfs_directory
from pyspark.ml.torch.distributor import TorchDistributor


def main():
    """Main execution function"""
    # Setup
    setup_logging()
    setup_environment()
    
    # Parse arguments and create configuration
    args = parse_args()
    config = Config(args)
    
    print("🚀 Starting GRU Model Training Pipeline")
    print(f"Mode: {config.mode}")
    print(f"Processes: {config.num_processes}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    
    # Initialize Spark
    spark = init_spark(config.mode)
    
    try:
        # Load dataset metadata
        config.num_features, config.num_classes, config.class_names = load_metadata(
            spark, config.train_path, config.label_indexer_path
        )
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.output_dir}_{timestamp}"
        config.output_dir = output_dir
        
        print(f"📁 Output directory: {output_dir}")
        
        # Prepare data partitions
        print("📊 Preparing data partitions...")
        
        # Validation data
        val_path = os.path.join(os.path.dirname(config.train_path), "val_df.parquet")
        config.val_temp_dir = f"hdfs://master:9000/tmp/val_data_{timestamp}"
        val_paths, val_lengths = prepare_data_partitions(
            spark, val_path, 1, config.val_temp_dir
        )
        validation_file_path = val_paths[0]
        
        # Training data
        config.train_temp_dir = f"hdfs://master:9000/tmp/train_data_{timestamp}"
        train_paths, train_lengths = prepare_data_partitions(
            spark, config.train_path, config.num_processes, config.train_temp_dir
        )
        
        if not any(train_paths):
            print("❌ Training data preparation failed.")
            return
        
        print(f"✅ Data preparation completed")
        print(f"   Training partitions: {sum(1 for p in train_paths if p)}/{len(train_paths)}")
        print(f"   Total training samples: {sum(train_lengths)}")
        print(f"   Validation samples: {sum(val_lengths)}")
        
        # Prepare arguments for distributed training
        training_args = config.to_dict()
        training_args.update({
            "partition_files": train_paths,
            "partition_lengths": train_lengths,
            "validation_file_path": validation_file_path,
            "timestamp": timestamp
        })
        
        # Run distributed training
        print("🏋️ Starting distributed training...")
        
        distributor = TorchDistributor(
            num_processes=config.num_processes,
            local_mode=(config.mode == "local"),
            use_gpu=False
        )
        
        training_result = distributor.run(training_function, args_dict=training_args)
        
        # Check training results
        if isinstance(training_result, dict) and training_result.get("status") == "SUCCESS":
            print("✅ Training completed successfully!")
            print(f"   Training time: {training_result.get('total_training_time', 0):.2f}s")
            print(f"   Best validation accuracy: {training_result.get('best_val_acc', 0):.4f}")
            
            # Evaluate on test set
            evaluation_result = evaluate_model_on_test_set(
                spark, config, output_dir, timestamp
            )
            
            if evaluation_result.get("status") == "SUCCESS":
                print("✅ Model evaluation completed!")
                
                # Combine results
                final_results = {
                    **training_result,
                    **evaluation_result,
                    "timestamp": timestamp
                }
                
                # Save comprehensive performance summary
                save_model_performance_summary(final_results, config, output_dir)
                
                print("\n🎉 Pipeline completed successfully!")
                print(f"📊 Results saved to: {output_dir}")
                
            else:
                print("⚠️ Model evaluation failed, but training was successful")
                
        else:
            print("❌ Training failed!")
            if isinstance(training_result, dict):
                print(f"Error: {training_result.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary directories
        print("🧹 Cleaning up temporary files...")
        
        cleanup_dirs = [
            config.train_temp_dir,
            config.val_temp_dir,
            f"hdfs://master:9000/tmp/test_data_{timestamp}" if 'timestamp' in locals() else None
        ]
        
        for temp_dir in cleanup_dirs:
            if temp_dir:
                cleanup_hdfs_directory(temp_dir)
        
        # Stop Spark session
        spark.stop()
        print("✅ Cleanup completed!")


if __name__ == "__main__":
    main()
