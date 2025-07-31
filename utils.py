import os
import json
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from urllib.parse import urlparse
import pyarrow.fs

def upload_local_directory_to_hdfs(local_path, hdfs_path):
    """Tải nội dung của một thư mục cục bộ lên HDFS."""
    try:
        print(f"[HDFS UPLOAD] From '{local_path}' to '{hdfs_path}'")
        parsed_uri = urlparse(hdfs_path)
        hdfs = pyarrow.fs.HadoopFileSystem(host=parsed_uri.hostname, port=parsed_uri.port)
        hdfs.create_dir(parsed_uri.path, recursive=True)
        for filename in os.listdir(local_path):
            local_file = os.path.join(local_path, filename)
            hdfs_file = os.path.join(parsed_uri.path, filename)
            if os.path.isfile(local_file):
                with open(local_file, 'rb') as f_local, hdfs.open_output_stream(hdfs_file) as f_hdfs:
                    f_hdfs.write(f_local.read())
    except Exception as e:
        print(f"[HDFS UPLOAD ERROR] {e}")

def save_and_upload_report(report_data, filename, output_dir):
    """Lưu một báo cáo (dictionary) thành tệp JSON và tải lên HDFS."""
    local_tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(local_tmp_dir, filename)
    try:
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o)
        print(f"[INFO] Saved {filename} locally to: {file_path}")
        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
    finally:
        shutil.rmtree(local_tmp_dir)

def plot_and_save_history(history, output_dir):
    """Vẽ đồ thị lịch sử huấn luyện và tải lên HDFS."""
    local_tmp_dir = tempfile.mkdtemp()
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], "b-o", label="Training Loss")
        ax1.plot(epochs, history['val_losses'], "r-o", label="Validation Loss")
        ax1.set_title("Model Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)
        ax2.plot(epochs, history['train_accuracies'], "b-o", label="Training Accuracy")
        ax2.plot(epochs, history['val_accuracies'], "r-o", label="Validation Accuracy")
        ax2.set_title("Model Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True); ax2.set_ylim([0, 1])
        plt.tight_layout()
        
        plot_path = os.path.join(local_tmp_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        history_path = os.path.join(local_tmp_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"[INFO] Saved history plots and data to: {local_tmp_dir}")
        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
    except Exception as e:
        print(f"[ERROR] Failed to plot history: {e}")
    finally:
        shutil.rmtree(local_tmp_dir)

def plot_and_save_confusion_matrix(cm, class_names, output_dir):
    """Vẽ và lưu ma trận nhầm lẫn dưới dạng heatmap."""
    try:
        local_tmp_dir = tempfile.mkdtemp()
        
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        figsize = (max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6))

        plt.figure(figsize=figsize)
        
        sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='coolwarm', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title("Confusion Matrix (%)", fontsize=14)
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_path = os.path.join(local_tmp_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Confusion matrix plot saved locally to {plot_path}")

        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
            print(f"[INFO] Successfully uploaded confusion_matrix.png to {output_dir}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to plot or save confusion matrix: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(local_tmp_dir)