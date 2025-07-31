#!/bin/bash
set -e

# -------------------------------------------------------------------
# KỊCH BẢN CHẠY ỨNG DỤNG HUẤN LUYỆN PHÂN TÁN
# -------------------------------------------------------------------
# Hướng dẫn:
# 1. Đặt các đường dẫn và tham số trong phần "CẤU HÌNH".
# 2. Chạy kịch bản từ terminal: ./run.sh
# -------------------------------------------------------------------

# --- CẤU HÌNH ---
# Đường dẫn đến môi trường Python trên các nút worker và driver
export PYSPARK_PYTHON=/home/ubuntu/spark_env/bin/python

# Các đường dẫn dữ liệu trên HDFS
TRAIN_DATA_PATH="hdfs://master:9000/usr/ubuntu/data/classweights-43-21/train_df.parquet"
TEST_DATA_PATH="hdfs://master:9000/usr/ubuntu/data/classweights-43-21/test_df.parquet"
LABEL_INDEXER_PATH="hdfs://master:9000/usr/ubuntu/data/classweights-43-21/data/label_indexer/label_indexer"

# Thư mục gốc trên HDFS để lưu kết quả
OUTPUT_DIR_BASE="hdfs://master:9000/usr/ubuntu/results-ultra"

# Các tham số cho Spark và ứng dụng
NUM_EXECUTORS=10
EXECUTOR_MEMORY="4g"
EXECUTOR_OVERHEAD="2g"
EXECUTOR_CORES=3
DRIVER_MEMORY="4g"

# Các siêu tham số cho mô hình
EPOCHS=5
BATCH_SIZE=512

# --- THỰC THI SPARK-SUBMIT ---

# Thêm các tệp Python của dự án để gửi đến các executor
PY_FILES="model.py,dataset.py,train.py,utils.py"

echo "Submitting Spark job..."

spark-submit \
--master yarn \
--deploy-mode cluster \
--name "GRU_Training_Refactored" \
--py-files $PY_FILES \
--num-executors $NUM_EXECUTORS \
--executor-memory $EXECUTOR_MEMORY \
--conf spark.executor.memoryOverhead=$EXECUTOR_OVERHEAD \
--executor-cores $EXECUTOR_CORES \
--driver-memory $DRIVER_MEMORY \
--conf spark.pyspark.python=$PYSPARK_PYTHON \
--conf spark.pyspark.driver.python=$PYSPARK_PYTHON \
--conf spark.executorEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON \
--conf spark.sql.adaptive.enabled=true \
--conf spark.sql.adaptive.coalescePartitions.enabled=true \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.network.timeout=800s \
--conf spark.executor.heartbeatInterval=60s \
--conf spark.task.maxFailures=4 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.scheduler.barrier.sync.timeout=600s \
--conf spark.rpc.askTimeout=400s \
--conf spark.python.worker.reuse=true \
--conf spark.kryo.unsafe=true \
--conf spark.default.parallelism=30 \
--conf spark.executor.extraJavaOptions="-XX:+UseG1GC" \
--conf spark.driver.extraJavaOptions="-XX:+UseG1GC" \
main.py \
--mode cluster \
--train_path $TRAIN_DATA_PATH \
--test_path $TEST_DATA_PATH \
--label_indexer_path $LABEL_INDEXER_PATH \
--output_dir $OUTPUT_DIR_BASE \
--epochs $EPOCHS \
--batch_size $BATCH_SIZE \
--num_processes $NUM_EXECUTORS

echo "Job submitted successfully."