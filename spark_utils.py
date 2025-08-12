"""
Spark session management utilities
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel


def init_spark(mode):
    """Initialize Spark Session based on execution mode"""
    if mode == "local":
        return (
            SparkSession.builder
            .appName("NF_UQ_NIDS_v2_Local_Training")
            .master("local[*]")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
    else:
        return (
            SparkSession.builder
            .appName("NF_UQ_NIDS_v2_Cluster_Training")
            .master("yarn")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )


def load_metadata(spark, train_path, label_indexer_path):
    """Load metadata about the dataset and labels"""
    try:
        # Load label indexer to get number of classes and class names
        label_indexer_model = StringIndexerModel.load(label_indexer_path)
        num_classes = len(label_indexer_model.labels)
        class_names = label_indexer_model.labels
        
        # Get number of features from first row
        first_row = spark.read.parquet(train_path).select("scaled_features").first()
        num_features = first_row["scaled_features"].size
        
        print(f"[METADATA] Features: {num_features}, Classes: {num_classes}")
        print(f"[METADATA] Class names: {class_names}")
        
        return num_features, num_classes, class_names
        
    except Exception as e:
        print(f"[WARNING] Error reading metadata: {e}. Using defaults.")
        return 39, 2, ["0", "1"]  # Default values
