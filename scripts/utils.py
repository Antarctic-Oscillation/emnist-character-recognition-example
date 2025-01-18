import os
import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_losses: list,
    train_accuracies: list,
    val_losses: list,
    val_accuracies: list,
    best_val_accuracy: float,
    is_best: bool = False,
    checkpoint_dir: str = 'checkpoints',
    latest_path: str = 'latest_model_checkpoint.pth',
    best_path: str = 'best_model_checkpoint.pth'
):
    """Save model checkpoint with training state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, latest_path))
    
    # Save best model separately if this is the best performance
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, best_path))

def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_dir: str = 'checkpoints',
    model_name: str = 'latest_model_checkpoint.pth'
) -> tuple:
    """Load latest checkpoint if available."""
    path = os.path.join(checkpoint_dir, model_name)
    
    if os.path.isfile(path):
        print("Loading latest checkpoint...")
        checkpoint = torch.load(path, weights_only=False)
        
        # Load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Return training state
        return (
            checkpoint['epoch'],
            checkpoint['train_losses'],
            checkpoint['train_accuracies'],
            checkpoint['val_losses'],
            checkpoint['val_accuracies'],
            checkpoint['best_val_accuracy']
        )
    
    return 0, [], [], [], [], 0.0