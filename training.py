"""
Training and evaluation logic for GRU model
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datetime import timedelta
from data_processing import create_pytorch_dataloader
from model import create_model
from hdfs_utils import upload_local_directory_to_hdfs


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    
    for features, labels, weights in dataloader:
        features, labels, weights = (
            features.to(device),
            labels.to(device),
            weights.to(device)
        )
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Apply sample weights
        weighted_loss = (loss * weights).mean()
        
        # Backward pass
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()


def evaluate_loop(model, dataloader, criterion, device):
    """Evaluate model on given dataloader"""
    if not dataloader:
        return 0.0, 0.0, [], []
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total_loss += loss.item() * features.size(0)
            total_correct += (predicted == labels).sum().item()
            total_samples += features.size(0)
            
            # Store predictions for detailed analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total_samples == 0:
        return 0.0, 0.0, [], []
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, all_labels, all_preds


def training_function(args_dict):
    """
    Main training function for distributed training
    This function will be executed on each worker process
    """
    try:
        # Get distributed training info
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize distributed training if needed
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", 
                init_method="env://", 
                timeout=timedelta(minutes=30)
            )
        
        print(f"[RANK {rank}] Starting training process")
        
        # Extract configuration
        num_features = args_dict["num_features"]
        num_classes = args_dict["num_classes"]
        batch_size = args_dict["batch_size"]
        max_epochs = args_dict["epochs"]
        learning_rate = args_dict["learning_rate"]
        hidden_size = args_dict["hidden_size"]
        dropout = args_dict["dropout"]
        
        # Create training dataloader for this rank
        partition_files = args_dict.get("partition_files", [])
        if rank >= len(partition_files) or not partition_files[rank]:
            print(f"[RANK {rank}] No data partition assigned")
            return {"status": "NO_DATA", "rank": rank}
        
        train_dataloader = create_pytorch_dataloader(
            partition_files[rank], batch_size, rank
        )
        
        if not train_dataloader:
            print(f"[RANK {rank}] Failed to create training dataloader")
            return {"status": "NO_DATA", "rank": rank}
        
        # Create validation dataloader (only for rank 0)
        val_dataloader = None
        if rank == 0 and args_dict.get("validation_file_path"):
            val_dataloader = create_pytorch_dataloader(
                args_dict["validation_file_path"], 
                batch_size * 2, 
                rank, 
                shuffle=False
            )
        
        # Setup device (CPU only for this configuration)
        device = torch.device("cpu")
        
        # Create model
        model = create_model_from_config(
            num_features, num_classes, hidden_size, dropout
        ).to(device)
        
        # Wrap model for distributed training if needed
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)
        
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(reduction="none")  # For weighted loss
        eval_criterion = nn.CrossEntropyLoss()  # For evaluation
        
        # Training metrics (only tracked on rank 0)
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        # Start training
        train_start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_epoch(model, train_dataloader, optimizer, criterion, device)
            
            # Synchronize all processes
            if world_size > 1:
                dist.barrier()
            
            # Validation and logging (only on rank 0)
            if rank == 0:
                val_start_time = time.time()
                
                # Get model for evaluation (unwrap DDP if needed)
                model_to_eval = model.module if world_size > 1 else model
                
                # Evaluate on validation set
                if val_dataloader:
                    val_loss, val_acc, _, _ = evaluate_loop(
                        model_to_eval, val_dataloader, eval_criterion, device
                    )
                else:
                    val_loss, val_acc = 0.0, 0.0
                
                val_duration = time.time() - val_start_time
                total_epoch_duration = time.time() - epoch_start_time
                train_duration = total_epoch_duration - val_duration
                
                # Store metrics
                train_losses.append(0)  # Not tracking train loss during training for efficiency
                train_accuracies.append(0)  # Not tracking train acc during training for efficiency
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Print progress
                print(f"--- EPOCH {epoch + 1}/{max_epochs} ---")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print(f"  Train Time: {train_duration:.2f}s | Val Time: {val_duration:.2f}s")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"🎉 New best validation accuracy: {best_val_acc:.4f}")
                    torch.save(model_to_eval.state_dict(), "/tmp/best_model.pth")
                
                # Save training history periodically
                if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
                    from reporting import plot_and_save_history
                    plot_and_save_history(
                        train_losses, train_accuracies, val_losses, val_accuracies,
                        args_dict["output_dir"]
                    )
        
        # Upload final model if it exists
        if rank == 0 and os.path.exists("/tmp/best_model.pth"):
            upload_local_directory_to_hdfs("/tmp/", args_dict["output_dir"])
        
        # Final synchronization
        if world_size > 1:
            dist.barrier()
        
        total_training_time = time.time() - train_start_time
        print(f"[RANK {rank}] Training completed in {total_training_time:.2f}s")
        
        return {
            "status": "SUCCESS",
            "rank": rank,
            "total_training_time": total_training_time,
            "best_val_acc": best_val_acc if rank == 0 else 0.0
        }
        
    except Exception as e:
        print(f"[TRAINING ERROR - RANK {rank}] {e}")
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "rank": rank, "message": str(e)}


def create_model_from_config(num_features, num_classes, hidden_size, dropout):
    """Create model from individual parameters"""
    from model import OptimizedGRUModel
    return OptimizedGRUModel(
        input_size=num_features,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout=dropout
    )
                