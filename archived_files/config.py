import os
import torch

# Configuration
DEBUG = True
CUTOFF_RADIUS = 5.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DIR = "../training_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Training Configuration
CONFIG = {
    'data_path': 'test_set_filtered.json',
    'seed': 42,
    'test_size': 0.2,
    'batch_size': 16,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 50,
    'save_path': 'best_model.pth'
}