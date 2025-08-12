"""
Data processing and loading utilities for distributed training
"""
import os
import pickle
import numpy as np
from urllib.parse import urlparse
import pyarrow.fs
import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array


def prepare_data_partitions(spark, data_path, num_processes, output_temp_dir):
    """
    Prepare data partitions for distributed training
    
    Args:
        spark: Spark session
        data_path: Path to input data (parquet)
        num_processes: Number of partitions to create
        output_temp_dir: Temporary directory to store partitions
    
    Returns:
        tuple: (partition_paths, partition_lengths)
    """
    print(f"[DATA PREP] Preparing data from {data_path} into {num_processes} partitions")
    print(f"[DATA PREP] Output temp dir: {output_temp_dir}")
    
    try:
        # Read and repartition data
        df = spark.read.parquet(data_path).repartition(num_processes)
        df = df.withColumn("features_array", vector_to_array(col("scaled_features")))
        
        def save_partition_and_get_len(partition_id, iterator):
            """Save partition data to HDFS and return metadata"""
            import pickle
            import numpy as np
            import os
            import pyarrow.fs
            from urllib.parse import urlparse
            
            BATCH_SIZE = 50000
            batch_data = []
            rows_processed = 0
            hdfs_path = os.path.join(output_temp_dir, f"partition_{partition_id}.pkl")
            
            try:
                parsed_uri = urlparse(hdfs_path)
                hdfs = pyarrow.fs.HadoopFileSystem(
                    host=parsed_uri.hostname, 
                    port=parsed_uri.port
                )
                
                with hdfs.open_output_stream(parsed_uri.path) as f:
                    for row in iterator:
                        try:
                            batch_data.append({
                                "features": np.array(row.features_array, dtype=np.float32),
                                "label": int(row.label),
                                "weight": float(row.weight)
                            })
                            rows_processed += 1
                            
                            # Write in batches to avoid memory issues
                            if len(batch_data) >= BATCH_SIZE:
                                pickle.dump(batch_data, f)
                                batch_data.clear()
                        except Exception as e:
                            print(f"[DATA PREP WARNING] Skipping row: {e}")
                            continue
                    
                    # Write remaining data
                    if batch_data:
                        pickle.dump(batch_data, f)
                
                if rows_processed > 0:
                    print(f"[DATA PREP] Partition {partition_id}: {rows_processed} rows -> {hdfs_path}")
                    return iter([(hdfs_path, rows_processed)])
                else:
                    print(f"[DATA PREP WARNING] Partition {partition_id}: No data")
                    return iter([])
                    
            except Exception as e:
                print(f"[DATA PREP ERROR] Partition {partition_id}: {e}")
                return iter([])
        
        # Process all partitions and collect results
        results = df.rdd.mapPartitionsWithIndex(save_partition_and_get_len).collect()
        
        # Initialize arrays for paths and lengths
        all_paths = [None] * num_processes
        all_lengths = [0] * num_processes
        
        # Populate arrays with results
        for path, length in results:
            try:
                part_id = int(path.split("_")[-1].split(".")[0])
                all_paths[part_id] = path
                all_lengths[part_id] = length
            except Exception as e:
                print(f"[DATA PREP WARNING] Error parsing partition ID: {e}")
                continue
        
        total_samples = sum(all_lengths)
        valid_partitions = sum(1 for p in all_paths if p is not None)
        
        print(f"[DATA PREP] Summary: {valid_partitions}/{num_processes} partitions, {total_samples} total samples")
        
        return all_paths, all_lengths
        
    except Exception as e:
        print(f"[DATA PREP ERROR] Failed to prepare data from {data_path}: {e}")
        return [None] * num_processes, [0] * num_processes


class InMemoryPartitionDataset(Dataset):
    """Dataset that loads partition data into memory"""
    
    def __init__(self, partition_file):
        self.data = []
        
        if not partition_file:
            print("[DATASET] No partition file provided")
            return
        
        try:
            if partition_file.startswith("hdfs://"):
                self._load_from_hdfs(partition_file)
            else:
                self._load_from_local(partition_file)
                
        except Exception as e:
            print(f"[DATASET ERROR] Failed to load {partition_file}: {e}")
    
    def _load_from_hdfs(self, partition_file):
        """Load data from HDFS partition file"""
        parsed_uri = urlparse(partition_file)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        
        with hdfs.open_input_stream(parsed_uri.path) as f:
            while True:
                try:
                    batch_data = pickle.load(f)
                    self.data.extend(batch_data)
                except EOFError:
                    break
    
    def _load_from_local(self, partition_file):
        """Load data from local partition file"""
        with open(partition_file, "rb") as f:
            while True:
                try:
                    batch_data = pickle.load(f)
                    self.data.extend(batch_data)
                except EOFError:
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.from_numpy(sample["features"]).float(),
            torch.tensor(sample["label"], dtype=torch.long),
            torch.tensor(sample["weight"], dtype=torch.float32)
        )


def create_pytorch_dataloader(partition_file, batch_size, rank, shuffle=True):
    """
    Create PyTorch DataLoader from partition file
    
    Args:
        partition_file: Path to partition file
        batch_size: Batch size for DataLoader
        rank: Process rank (for logging)
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader or None if no data
    """
    dataset = InMemoryPartitionDataset(partition_file)
    
    if len(dataset) == 0:
        print(f"[RANK {rank}] Empty dataset from {partition_file}")
        return None
    
    print(f"[RANK {rank}] Created DataLoader with {len(dataset)} samples from {partition_file}")
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        shuffle=shuffle
    )


def get_data_statistics(dataloader):
    """Get basic statistics about the data"""
    if not dataloader:
        return {}
    
    total_samples = 0
    label_counts = {}
    
    for features, labels, weights in dataloader:
        total_samples += features.size(0)
        
        for label in labels.numpy():
            label_counts[int(label)] = label_counts.get(int(label), 0) + 1
    
    return {
        "total_samples": total_samples,
        "label_distribution": label_counts,
        "num_classes": len(label_counts)
    }
