import os
import argparse
import tempfile
import shutil
from datetime import datetime
from urllib.parse import urlparse
import torch
import torch.nn as nn
import pyarrow.fs
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.torch.distributor import TorchDistributor
from sklearn.metrics import classification_report, confusion_matrix

from train import training_function, evaluate_loop
from dataset import prepare_data_partitions, create_pytorch_dataloader
from model import OptimizedGRUModel
from utils import save_and_upload_report, plot_and_save_confusion_matrix

def init_spark(mode):
    """Khởi tạo Spark Session."""
    if mode == "local":
        return SparkSession.builder.appName("Local_Training").master("local[*]").getOrCreate()
    else:
        return SparkSession.builder.appName("Cluster_Training").master("yarn").getOrCreate()

def parse_args():
    """Phân tích các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Distributed GRU Model Training")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "cluster"], help="Execution mode")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data parquet file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data parquet file")
    parser.add_argument("--label_indexer_path", type=str, required=True, help="Path to the fitted StringIndexerModel")
    parser.add_argument("--output_dir", type=str, required=True, help="HDFS directory to save results and models")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--num_processes", type=int, default=2, help="Number of distributed processes")
    return parser.parse_args()

def main():
    """Điểm vào chính của ứng dụng."""
    args = parse_args()
    spark = init_spark(args.mode)
    args_dict = vars(args)

    try:
        label_indexer_model = StringIndexerModel.load(args.label_indexer_path)
        args_dict["num_classes"] = len(label_indexer_model.labels)
        args_dict["class_names"] = label_indexer_model.labels
        first_row = spark.read.parquet(args.train_path).select("scaled_features").first()
        args_dict["num_features"] = first_row["scaled_features"].size
    except Exception as e:
        print(f"Error reading metadata: {e}. Using defaults.")
        args_dict["num_classes"], args_dict["class_names"], args_dict["num_features"] = 2, ["0","1"], 39

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    args_dict["output_dir"] = output_dir

    val_path = os.path.join(os.path.dirname(args.train_path), "val_df.parquet")
    val_temp_dir = f"hdfs://master:9000/tmp/val_data_{timestamp}"
    val_paths, _ = prepare_data_partitions(spark, val_path, 1, val_temp_dir)
    args_dict["validation_file_path"] = val_paths[0]

    train_temp_dir = f"hdfs://master:9000/tmp/train_data_{timestamp}"
    train_paths, _ = prepare_data_partitions(spark, args.train_path, args.num_processes, train_temp_dir)
    args_dict["partition_files"] = train_paths

    if not any(args_dict["partition_files"]):
        print("[ERROR] Training data preparation failed."); spark.stop(); return

    distributor = TorchDistributor(num_processes=args.num_processes, local_mode=(args.mode=="local"), use_gpu=False)
    result = distributor.run(training_function, args_dict=args_dict)

    if isinstance(result, dict) and result.get("status") == "SUCCESS":
        print(f"🔍 Starting FINAL evaluation on TEST set...")
        try:
            model_path_hdfs = os.path.join(output_dir, "best_model.pth")
            print(f"Loading BEST model for final evaluation from {model_path_hdfs}")
            
            model = OptimizedGRUModel(args_dict["num_features"], args_dict["num_classes"])
            
            parsed_uri = urlparse(model_path_hdfs)
            hdfs = pyarrow.fs.HadoopFileSystem(host=parsed_uri.hostname, port=parsed_uri.port)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                with hdfs.open_input_stream(parsed_uri.path) as hdfs_f:
                    tmp_f.write(hdfs_f.read())
                model.load_state_dict(torch.load(tmp_f.name))
            os.remove(tmp_f.name)
            
            model.to(torch.device("cpu"))
            print("Best model loaded successfully for final evaluation.")

            test_temp_dir = f"hdfs://master:9000/tmp/test_data_{timestamp}"
            test_paths, _ = prepare_data_partitions(spark, args.test_path, 1, test_temp_dir)
            test_dataloader = create_pytorch_dataloader(test_paths[0], args.batch_size * 2, shuffle=False)
            
            if test_dataloader:
                _, test_acc, all_labels, all_preds = evaluate_loop(model, test_dataloader, nn.CrossEntropyLoss(), torch.device("cpu"))
                print(f"\n--- FINAL TEST SET PERFORMANCE ---")
                print(f"  Test Accuracy: {test_acc:.4f}")
                report = classification_report(all_labels, all_preds, target_names=args_dict["class_names"], output_dict=True, zero_division=0)
                save_and_upload_report(report, "final_test_report.json", output_dir)

                cm = confusion_matrix(all_labels, all_preds)
                plot_and_save_confusion_matrix(cm, args_dict["class_names"], output_dir)

        except Exception as e:
            print(f"[ERROR] Could not perform final evaluation: {e}")

    temp_dirs_to_clean = [train_temp_dir, val_temp_dir]
    if 'test_temp_dir' in locals(): temp_dirs_to_clean.append(test_temp_dir)
    
    for temp_dir in temp_dirs_to_clean:
        try:
            parsed_uri = urlparse(temp_dir)
            hdfs = pyarrow.fs.HadoopFileSystem(host=parsed_uri.hostname, port=parsed_uri.port)
            if hdfs.get_file_info(parsed_uri.path).type != pyarrow.fs.FileType.NotFound:
                hdfs.delete_dir(temp_dir)
                print(f"[CLEANUP] Removed temp HDFS directory: {temp_dir}")
        except Exception as e: 
            print(f"[CLEANUP] Failed to remove {temp_dir}: {e}")
    
    spark.stop()
    print("Pipeline completed!")

if __name__ == "__main__":
    main()