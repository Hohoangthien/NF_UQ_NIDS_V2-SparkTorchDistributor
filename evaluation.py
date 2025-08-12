"""
Model evaluation utilities
"""
import os
import tempfile
import torch
import torch.nn as nn
from urllib.parse import urlparse
import pyarrow.fs
from sklearn.metrics import confusion_matrix
from data_processing import create_pytorch_dataloader
from model import OptimizedGRUModel
from training import evaluate_loop
from reporting import (
    plot_and_save_confusion_matrix, 
    generate_classification_report,
    save_model_performance_summary
)


def load_model_from_hdfs(model_path_hdfs, num_features, num_classes, hidden_size=64, dropout=0.2):
    """Load trained model from HDFS"""
    try:
        print(f"[EVALUATION] Loading model from {model_path_hdfs}")
        
        # Create model instance
        model = OptimizedGRUModel(
            input_size=num_features,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Download model state from HDFS
        parsed_uri = urlparse(model_path_hdfs)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
            with hdfs.open_input_stream(parsed_uri.path) as hdfs_f:
                tmp_f.write(hdfs_f.read())
            
            # Load state dict
            model.load_state_dict(torch.load(tmp_f.name, map_location='cpu'))
        
        # Clean up temporary file
        os.remove(tmp_f.name)
        
        model.to(torch.device("cpu"))
        print("[EVALUATION] Model loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"[EVALUATION ERROR] Failed to load model: {e}")
        raise


def evaluate_model_on_test_set(spark, config, output_dir, timestamp):
    """Evaluate the trained model on test set"""
    try:
        print("🔍 Starting final evaluation on test set...")
        
        # Prepare test data
        from data_processing import prepare_data_partitions
        test_temp_dir = f"hdfs://master:9000/tmp/test_data_{timestamp}"
        test_paths, _ = prepare_data_partitions(
            spark, config.test_path, 1, test_temp_dir
        )
        
        if not test_paths[0]:
            print("[EVALUATION ERROR] Failed to prepare test data")
            return {"status": "ERROR", "message": "Test data preparation failed"}
        
        # Create test dataloader
        test_dataloader = create_pytorch_dataloader(
            test_paths[0], 
            config.batch_size * 2, 
            rank=0, 
            shuffle=False
        )
        
        if not test_dataloader:
            print("[EVALUATION ERROR] Failed to create test dataloader")
            return {"status": "ERROR", "message": "Test dataloader creation failed"}
        
        # Load trained model
        model_path_hdfs = os.path.join(output_dir, "best_model.pth")
        model = load_model_from_hdfs(
            model_path_hdfs,
            config.num_features,
            config.num_classes,
            config.hidden_size,
            config.dropout
        )
        
        # Evaluate model
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc, all_labels, all_preds = evaluate_loop(
            model, test_dataloader, criterion, device
        )
        
        print(f"\n--- FINAL TEST SET PERFORMANCE ---")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Generate detailed reports
        report = generate_classification_report(
            all_labels, all_preds, config.class_names, output_dir
        )
        
        # Create and save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_and_save_confusion_matrix(cm, config.class_names, output_dir)
        
        # Clean up test data
        from hdfs_utils import cleanup_hdfs_directory
        cleanup_hdfs_directory(test_temp_dir)
        
        return {
            "status": "SUCCESS",
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
    except Exception as e:
        print(f"[EVALUATION ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "message": str(e)}


def evaluate_model_performance(model, dataloader, device, class_names):
    """
    Comprehensive model performance evaluation
    
    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names
    
    Returns:
        dict: Comprehensive evaluation results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = total_loss / total_samples
    
    # Generate classification report
    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probabilities,
        "classification_report": report,
        "confusion_matrix": cm,
        "total_samples": total_samples
    }


def compare_model_performance(results_list, model_names, output_dir):
    """Compare performance of multiple models"""
    try:
        comparison_data = {
            "models": model_names,
            "comparison": {}
        }
        
        metrics_to_compare = ["accuracy", "loss"]
        
        for metric in metrics_to_compare:
            comparison_data["comparison"][metric] = [
                results[metric] for results in results_list
            ]
        
        # Add classification report metrics
        for avg_type in ["macro avg", "weighted avg"]:
            for metric in ["precision", "recall", "f1-score"]:
                key = f"{avg_type}_{metric}"
                comparison_data["comparison"][key] = [
                    results["classification_report"][avg_type][metric] 
                    for results in results_list
                ]
        
        # Save comparison
        from reporting import save_and_upload_report
        save_and_upload_report(
            comparison_data, 
            "model_comparison.json", 
            output_dir
        )
        
        # Print comparison summary
        print("\n--- MODEL COMPARISON ---")
        for i, model_name in enumerate(model_names):
            print(f"{model_name}:")
            print(f"  Accuracy: {comparison_data['comparison']['accuracy'][i]:.4f}")
            print(f"  Loss: {comparison_data['comparison']['loss'][i]:.4f}")
            print(f"  F1-Score (Macro): {comparison_data['comparison']['macro avg_f1-score'][i]:.4f}")
        
        return comparison_data
        
    except Exception as e:
        print(f"[COMPARISON ERROR] {e}")
        return {}
