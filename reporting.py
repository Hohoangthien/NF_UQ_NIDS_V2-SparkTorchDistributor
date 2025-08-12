"""
Reporting and visualization utilities
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from hdfs_utils import create_temp_local_dir, cleanup_local_dir, upload_local_directory_to_hdfs
import os


def save_and_upload_report(report_data, filename, output_dir):
    """Save report data to JSON and upload to HDFS"""
    local_tmp_dir = create_temp_local_dir()
    
    try:
        file_path = os.path.join(local_tmp_dir, filename)
        
        with open(file_path, "w") as f:
            json.dump(
                report_data,
                f,
                indent=2,
                default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else o
            )
        
        print(f"[REPORT] Saved {filename} locally to: {file_path}")
        
        # Upload to HDFS if needed
        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
            print(f"[REPORT] Uploaded {filename} to {output_dir}")
        
    except Exception as e:
        print(f"[REPORT ERROR] Failed to save {filename}: {e}")
    finally:
        cleanup_local_dir(local_tmp_dir)


def plot_and_save_history(train_losses, train_accuracies, val_losses, val_accuracies, output_dir):
    """Plot and save training history"""
    try:
        local_tmp_dir = create_temp_local_dir()
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, "b-o", label="Training Loss", markersize=4)
        ax1.plot(epochs, val_losses, "r-o", label="Validation Loss", markersize=4)
        ax1.set_title("Model Loss", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accuracies, "b-o", label="Training Accuracy", markersize=4)
        ax2.plot(epochs, val_accuracies, "r-o", label="Validation Accuracy", markersize=4)
        ax2.set_title("Model Accuracy", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(local_tmp_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[REPORT] Training history plot saved to {plot_path}")
        
        # Save history data as JSON
        history_data = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "epochs": len(train_losses)
        }
        
        history_path = os.path.join(local_tmp_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        
        # Upload to HDFS if needed
        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
            print(f"[REPORT] Training history uploaded to {output_dir}")
        
    except Exception as e:
        print(f"[REPORT ERROR] Failed to plot training history: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_local_dir(local_tmp_dir)


def plot_and_save_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix as heatmap"""
    try:
        local_tmp_dir = create_temp_local_dir()
        
        # Calculate percentage confusion matrix
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Determine figure size based on number of classes
        figsize = (max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6))
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm_percentage, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title("Confusion Matrix (%)", fontsize=14)
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(local_tmp_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[REPORT] Confusion matrix saved to {plot_path}")
        
        # Save confusion matrix data
        cm_data = {
            "confusion_matrix_counts": cm.tolist(),
            "confusion_matrix_percentages": cm_percentage.tolist(),
            "class_names": class_names
        }
        
        cm_json_path = os.path.join(local_tmp_dir, "confusion_matrix.json")
        with open(cm_json_path, "w") as f:
            json.dump(cm_data, f, indent=2)
        
        # Upload to HDFS if needed
        if output_dir.startswith("hdfs://"):
            upload_local_directory_to_hdfs(local_tmp_dir, output_dir)
            print(f"[REPORT] Confusion matrix uploaded to {output_dir}")
        
    except Exception as e:
        print(f"[REPORT ERROR] Failed to create confusion matrix: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_local_dir(local_tmp_dir)


def generate_classification_report(y_true, y_pred, class_names, output_dir):
    """Generate and save detailed classification report"""
    try:
        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Add additional metrics
        report["total_samples"] = len(y_true)
        report["num_classes"] = len(class_names)
        
        # Calculate per-class sample counts
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        report["class_distribution"] = {
            class_names[i]: class_distribution.get(i, 0) 
            for i in range(len(class_names))
        }
        
        # Save report
        save_and_upload_report(report, "classification_report.json", output_dir)
        
        # Print summary
        print("\n--- CLASSIFICATION REPORT SUMMARY ---")
        print(f"Total Samples: {report['total_samples']}")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return report
        
    except Exception as e:
        print(f"[REPORT ERROR] Failed to generate classification report: {e}")
        return {}


def save_model_performance_summary(results, config, output_dir):
    """Save overall model performance summary"""
    try:
        summary = {
            "experiment_info": {
                "timestamp": results.get("timestamp"),
                "mode": config.mode,
                "num_processes": config.num_processes
            },
            "model_config": {
                "num_features": config.num_features,
                "num_classes": config.num_classes,
                "hidden_size": config.hidden_size,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs
            },
            "training_results": {
                "status": results.get("status"),
                "total_training_time": results.get("total_training_time"),
                "best_validation_accuracy": results.get("best_val_acc")
            },
            "test_results": {
                "test_accuracy": results.get("test_accuracy"),
                "test_loss": results.get("test_loss")
            }
        }
        
        save_and_upload_report(summary, "experiment_summary.json", output_dir)
        
        print("\n--- EXPERIMENT SUMMARY ---")
        print(f"Status: {summary['training_results']['status']}")
        print(f"Training Time: {summary['training_results']['total_training_time']:.2f}s")
        print(f"Best Val Accuracy: {summary['training_results']['best_validation_accuracy']:.4f}")
        print(f"Test Accuracy: {summary['test_results']['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"[REPORT ERROR] Failed to save experiment summary: {e}")
