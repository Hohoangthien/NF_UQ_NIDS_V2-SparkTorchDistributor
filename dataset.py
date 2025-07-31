import os
import pickle
import numpy as np
from urllib.parse import urlparse
import pyarrow.fs
import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.sql.functions import col, vector_to_array

class InMemoryPartitionDataset(Dataset):
    def __init__(self, partition_file):
        self.data = []
        if not partition_file:
            return
        try:
            if partition_file.startswith("hdfs://"):
                parsed_uri = urlparse(partition_file)
                hdfs = pyarrow.fs.HadoopFileSystem(
                    host=parsed_uri.hostname, port=parsed_uri.port
                )
                with hdfs.open_input_stream(parsed_uri.path) as f:
                    while True:
                        try:
                            self.data.extend(pickle.load(f))
                        except EOFError:
                            break
            else:
                with open(partition_file, "rb") as f:
                    while True:
                        try:
                            self.data.extend(pickle.load(f))
                        except EOFError:
                            break
        except Exception as e:
            print(f"[DATASET_ERROR] Failed to load {partition_file}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.from_numpy(sample["features"]).float(),
            torch.tensor(sample["label"], dtype=torch.long),
            torch.tensor(sample["weight"], dtype=torch.float32),
        )

def create_pytorch_dataloader(partition_file, batch_size, rank=0, shuffle=True):
    """Tạo một DataLoader từ một đường dẫn tệp phân vùng."""
    dataset = InMemoryPartitionDataset(partition_file)
    if len(dataset) == 0:
        print(f"[RANK {rank}] Empty dataset from {partition_file}")
        return None
    print(f"[RANK {rank}] Created InMemory DataLoader with {len(dataset)} samples.")
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

def prepare_data_partitions(spark, data_path, num_processes, output_temp_dir):
    """Chuẩn bị và lưu các phân vùng dữ liệu vào HDFS."""
    print(f"[DRIVER] Preparing data from {data_path} into {num_processes} partitions -> {output_temp_dir}")
    try:
        df = spark.read.parquet(data_path).repartition(num_processes)
        df = df.withColumn("features_array", vector_to_array(col("scaled_features")))

        def save_partition_and_get_len(partition_id, iterator):
            import pickle, numpy as np, os, pyarrow.fs
            from urllib.parse import urlparse

            BATCH_SIZE = 50000
            batch_data = []
            rows_processed = 0
            hdfs_path = os.path.join(output_temp_dir, f"partition_{partition_id}.pkl")
            try:
                parsed_uri = urlparse(hdfs_path)
                hdfs = pyarrow.fs.HadoopFileSystem(
                    host=parsed_uri.hostname, port=parsed_uri.port
                )
                with hdfs.open_output_stream(parsed_uri.path) as f:
                    for row in iterator:
                        try:
                            batch_data.append(
                                {
                                    "features": np.array(
                                        row.features_array, dtype=np.float32
                                    ),
                                    "label": int(row.label),
                                    "weight": float(row.weight),
                                }
                            )
                            rows_processed += 1
                            if len(batch_data) >= BATCH_SIZE:
                                pickle.dump(batch_data, f)
                                batch_data.clear()
                        except Exception:
                            continue
                    if batch_data:
                        pickle.dump(batch_data, f)

                if rows_processed > 0:
                    return iter([(hdfs_path, rows_processed)])
                else:
                    return iter([])
            except Exception:
                return iter([])

        results = df.rdd.mapPartitionsWithIndex(save_partition_and_get_len).collect()

        all_paths = [None] * num_processes
        all_lengths = [0] * num_processes
        for path, length in results:
            try:
                part_id = int(path.split("_")[-1].split(".")[0])
                all_paths[part_id] = path
                all_lengths[part_id] = length
            except:
                pass
        return all_paths, all_lengths
    except Exception as e:
        print(f"[DRIVER] Data preparation failed for {data_path}: {e}")
        return [None] * num_processes, [0] * num_processes