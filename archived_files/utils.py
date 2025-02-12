import torch
import matplotlib.pyplot as plt

from archived_files.config import DEBUG, PLOT_DIR


def debug_print(title, data=None, tensors=None, stats=False):
    """Prints debug information if DEBUG is enabled."""
    if DEBUG:
        separator = "\n" + "=" * 80
        print(separator)
        print(f"DEBUG: {title}")

        if data is not None:
            print("- Data Details:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {type(v)} | Content: {v}")
            else:
                print(f"- Type: {type(data)}")
                print(f"- Content: {data}")

        if tensors is not None:
            print("- Tensor Details:")
            if isinstance(tensors, dict):
                for name, tensor in tensors.items():
                    _print_tensor_info(name, tensor, stats)
            else:
                _print_tensor_info("Tensor", tensors, stats)

        print(separator + "\n")


def _print_tensor_info(name, tensor, stats=False):
    """Prints details about a tensor."""
    if tensor is None:
        print(f"  {name}: None")
        return

    print(f"  {name}:")
    print(f"    Shape: {tensor.shape}")
    print(f"    Dtype: {tensor.dtype}")
    print(f"    Device: {tensor.device}")

    if stats and tensor.numel() > 0:
        t = tensor.detach().cpu().float()
        print(f"    Min: {t.min().item():.4f}")
        print(f"    Max: {t.max().item():.4f}")
        print(f"    Mean: {t.mean().item():.4f}")
        print(f"    Std: {t.std().item():.4f}")
        print(f"    NaN: {torch.isnan(t).any().item()}")
        print(f"    Inf: {torch.isinf(t).any().item()}")
    else:
        if tensor.numel() > 10:
            print(f"    Values (first 10): {tensor.flatten()[:10]}")
        else:
            print(f"    Values: {tensor}")


def plot_loss_curves(train_losses, val_losses, epoch):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curves - Epoch {epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/loss_epoch_{epoch}.png")
    plt.close()