import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datetime import timedelta
import tempfile
import shutil
import time

from model import OptimizedGRUModel
from dataset import create_pytorch_dataloader
from utils import plot_and_save_history, upload_local_directory_to_hdfs

def evaluate_loop(model, dataloader, criterion, device):
    """Vòng lặp đánh giá chung cho validation và test."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total_loss += loss.item() * features.size(0)
            total_correct += (predicted == labels).sum().item()
            total_samples += features.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if total_samples == 0: return 0.0, 0.0, [], []
    return total_loss / total_samples, total_correct / total_samples, all_labels, all_preds

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Một epoch huấn luyện."""
    model.train()
    for features, labels, weights in dataloader:
        features, labels, weights = features.to(device), labels.to(device), weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

def training_function(args_dict):
    """Hàm chính được thực thi bởi mỗi tiến trình PyTorch."""
    try:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method="env://", timeout=timedelta(minutes=30))

        num_features, num_classes, batch_size, max_epochs = (
            args_dict["num_features"], args_dict["num_classes"],
            args_dict["batch_size"], args_dict["epochs"],
        )

        train_dataloader = create_pytorch_dataloader(args_dict["partition_files"][rank], batch_size, rank, shuffle=True)
        if not train_dataloader:
            return {"status": "NO_DATA", "rank": rank}

        val_dataloader = None
        if rank == 0 and args_dict.get("validation_file_path"):
            val_dataloader = create_pytorch_dataloader(args_dict["validation_file_path"], batch_size * 2, rank, shuffle=False)

        device = torch.device("cpu")
        model = OptimizedGRUModel(num_features, num_classes).to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)

        optimizer = optim.AdamW(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss(reduction='none')
        eval_criterion = nn.CrossEntropyLoss()

        history = {"train_losses": [], "train_accuracies": [], "val_losses": [], "val_accuracies": []}
        best_val_acc = 0.0
        
        train_start_time = time.time()
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            train_epoch(model, train_dataloader, optimizer, criterion, device)
            if world_size > 1: dist.barrier()

            if rank == 0:
                val_start_time = time.time()
                model_to_eval = model.module if world_size > 1 else model
                val_loss, val_acc, _, _ = evaluate_loop(model_to_eval, val_dataloader, eval_criterion, device) if val_dataloader else (0, 0, [], [])
                val_duration_seconds = time.time() - val_start_time

                history["train_losses"].append(0)
                history["train_accuracies"].append(0)
                history["val_losses"].append(val_loss)
                history["val_accuracies"].append(val_acc)

                total_epoch_duration = time.time() - epoch_start_time
                train_duration_seconds = total_epoch_duration - val_duration_seconds

                print(f"--- EPOCH {epoch + 1}/{max_epochs} ---")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
                print(f"  Distributed Train Time: {train_duration_seconds:.2f}s | Validation Time: {val_duration_seconds:.2f}s")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"🎉 New best val_acc: {best_val_acc:.4f}. Saving... ")
                    torch.save(model_to_eval.state_dict(), "/tmp/best_model.pth")
                
                plot_and_save_history(history, args_dict["output_dir"])

        if rank == 0 and os.path.exists("/tmp/best_model.pth"):
            model_tmp_dir = tempfile.mkdtemp()
            shutil.move("/tmp/best_model.pth", os.path.join(model_tmp_dir, "best_model.pth"))
            upload_local_directory_to_hdfs(model_tmp_dir, args_dict["output_dir"])
            shutil.rmtree(model_tmp_dir)

        if world_size > 1: dist.barrier()
        
        train_duration_time = time.time() - train_start_time
        print(f"  Total Train Time: {train_duration_time:.2f}s")
        
        return {"status": "SUCCESS", "Total_training_time": train_duration_time}

    except Exception as e:
        import traceback
        print(f"[TRAINING_FUNC_ERROR] {e}")
        traceback.print_exc()
        return {"status": "ERROR", "message": str(e)}