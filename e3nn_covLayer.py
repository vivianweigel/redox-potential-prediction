import os
import json
from io import StringIO
from typing import Dict, List, Optional
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from ase.io import read
from ase import Atoms
import seaborn as sns
from scipy import stats

from e3nn import o3
from e3nn.nn import FullyConnectedNet

warnings.filterwarnings('ignore', module='ase')
DEBUG = "OFF"

def debug_print(step: str, msg: str):
    """Prints a debug message with a prefix indicating the code stage."""
    if DEBUG == "ON":
        print(f"[DEBUG - {step}] {msg}")
    elif DEBUG == "OFF":
        pass

class MolecularGraph:
    """
    Constructs a PyG Data object from ase.Atoms.
    """
    def __init__(self, cutoff: float = 5.0):
        self.cutoff = cutoff

    def build_molecular_graph(self, atoms: Atoms,
                              type_encoding: Dict[str, int],
                              type_onehot: torch.Tensor) -> Data:
        debug_print("MolecularGraph", "Building molecular graph...")

        # Nodes: one-hot encoding based on atomic symbol
        x_list = []
        for sym in atoms.get_chemical_symbols():
            x_list.append(type_onehot[type_encoding[sym]])
        x = torch.stack(x_list, dim=0)

        # Atomic positions
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        # Neighbor list
        from ase.neighborlist import neighbor_list
        i_idx, j_idx, S = neighbor_list("ijS", atoms, self.cutoff)

        # edge_index and edge_vec
        edge_index = torch.stack([torch.LongTensor(i_idx), torch.LongTensor(j_idx)], dim=0)
        edge_vec = pos[j_idx] - pos[i_idx]

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index
        )
        data.edge_vec = edge_vec

        debug_print(
            "MolecularGraph",
            f"Graph with {data.num_nodes} atoms and {data.edge_index.shape[1]} bonds."
        )
        return data

class MolecularDataset(Dataset):
    def __init__(self, database_path: str = 'fd.json', cutoff: float = 5.0, test_size: float = 0.2,
                 required_fields=None):
        super().__init__()
        debug_print("MolecularDataset.__init__", f"Reading {database_path}")

        self.cutoff = cutoff
        self.required_fields = required_fields if required_fields else ["opt_molecule_S0", "reduction_potential_S1 (eV)"]

        if not os.path.exists(database_path):
            raise FileNotFoundError(f"File '{database_path}' not found.")

        with open(database_path, 'r') as f:
            raw_data = json.load(f)

        # Filter dataset to remove entries with missing fields or None/null values
        self.database = [
            entry for entry in raw_data
            if all(field in entry and entry[field] is not None for field in self.required_fields)
        ]

        debug_print("MolecularDataset.__init__",
                    f"Filtered dataset size: {len(self.database)} (removed {len(raw_data) - len(self.database)} entries)")

        if len(self.database) == 0:
            raise ValueError("No valid data after filtering. Check the dataset or required fields.")

        # Collect atomic symbols
        all_symbols = set()
        for entry in self.database:
            xyz_string = entry["opt_molecule_S0"]
            atoms = read(StringIO(xyz_string), format='xyz')
            all_symbols.update(atoms.get_chemical_symbols())

        # Symbol encoding
        self.type_encoding = {sym: i for i, sym in enumerate(sorted(all_symbols))}
        self.type_onehot = torch.eye(len(self.type_encoding))

        # Train-test split
        self.train_idx, self.test_idx = train_test_split(
            range(len(self.database)), test_size=test_size, random_state=42
        )

        print(f"[INFO] Train set size: {len(self.train_idx)}, Test set size: {len(self.test_idx)}")

        # Graph generator
        self.graph_builder = MolecularGraph(cutoff=self.cutoff)

        debug_print(
            "MolecularDataset.__init__",
            f"Atomic symbols: {len(self.type_encoding)}. Train: {len(self.train_idx)}, Test: {len(self.test_idx)}"
        )

    def len(self):
        return len(self.database)

    def get(self, idx: int) -> Data:
        entry = self.database[idx]
        debug_print("MolecularDataset.get", f"Item {idx}")

        xyz_string = entry["opt_molecule_S0"]
        atoms = read(StringIO(xyz_string), format='xyz')
        homo_value = entry["homo (eV)"]

        data = self.graph_builder.build_molecular_graph(
            atoms=atoms,
            type_encoding=self.type_encoding,
            type_onehot=self.type_onehot
        )
        homo_tensor = torch.tensor([homo_value], dtype=torch.float32).view(-1, 1)  # Shape: [n_atoms, 1]
        data.x = torch.cat([data.x, homo_tensor.repeat(data.x.size(0), 1)],
                           dim=-1)  # Concatenate HOMO to each atom's featurev
        data.y = torch.tensor([entry["reduction_potential_S1 (eV)"]], dtype=torch.float32)
        return data

    @property
    def train_dataset(self):
        debug_print("MolecularDataset.train_dataset", "Accessing TRAIN dataset.")
        return [self.get(i) for i in self.train_idx]

    @property
    def test_dataset(self):
        debug_print("MolecularDataset.test_dataset", "Accessing TEST dataset.")
        return [self.get(i) for i in self.test_idx]

# =============== (3) Message passing layer (simplified) ===============
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        debug_print("ConvLayer.__init__",
                    f"Creating ConvLayer with in={in_channels}, out={out_channels}")

        self.node_mlp = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Linear(1, out_channels)

    def forward(self, node_features, edge_index, edge_vec):
        debug_print("ConvLayer.forward", "Forward pass in ConvLayer.")
        row, col = edge_index
        edge_length = edge_vec.norm(dim=1, keepdim=True)

        node_emb = self.node_mlp(node_features)
        edge_emb = self.edge_mlp(edge_length)

        messages = node_emb[row] * edge_emb
        aggregated = scatter(messages, col, dim=0, reduce='add')

        return aggregated

# =============== (4) E3NN-like network (simplified) ===============
class MolecularE3NN(nn.Module):
    def __init__(self, num_atom_types: int, hidden_channels: int = 64,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        debug_print("MolecularE3NN.__init__",
                    f"Creating MolecularE3NN with hidden={hidden_channels}, layers={num_layers}")

        self.embedding = nn.Linear(num_atom_types + 1, hidden_channels)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                ConvLayer(hidden_channels, hidden_channels)
            )

        self.dropout = nn.Dropout(dropout)

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        debug_print("MolecularE3NN.forward", "Starting forward pass in the model.")
        x = self.embedding(data.x)

        for conv in self.conv_layers:
            x_new = conv(x, data.edge_index, data.edge_vec)
            x = x + self.dropout(x_new)  # residual

        batch = data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x_concat = torch.cat([x_mean, x_max], dim=-1)
        out = self.readout_mlp(x_concat)

        return out.squeeze(-1)

# =============== (5) Training function ===============
def train_model(dataset, num_epochs=10, batch_size=2, device='cpu', patience=5):
    debug_print("train_model", "Starting training.")
    model = MolecularE3NN(
        num_atom_types=len(dataset.type_encoding),
        hidden_channels=64,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = nn.L1Loss()

    train_loader = tg.loader.DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = tg.loader.DataLoader(dataset.test_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    no_improve = 0

    # Lists to store losses
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        debug_print("train_model", f"=== Epoch {epoch+1}/{num_epochs} ===")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if not hasattr(batch, 'batch'):
                batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)

            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        if len(train_loader) > 0:
            train_loss /= len(train_loader)

        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if not hasattr(batch, 'batch'):
                    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
                batch = batch.to(device)
                pred = model(batch)
                test_loss += criterion(pred, batch.y).item()

        if len(test_loader) > 0:
            test_loss /= len(test_loader)

        scheduler.step(test_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Save losses in lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # # Early stopping
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     no_improve = 0
        #
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    torch.save(model.state_dict(), 'trained_model.pt')

    debug_print("train_model", f"Best Test Loss: {best_loss:.4f}. Loading model.")
    model.load_state_dict(torch.load('trained_model.pt', map_location=device))

    # Plot metrics at the end
    plot_metrics(train_losses, test_losses)

    return model

# =============== (6) Evaluate on the split test set ===============
def evaluate_model(model, dataset, device='cpu'):
    """
    Evaluates the model on the test subset of the dataset split with train_test_split.
    Returns predictions, true values, and average MAE.
    """
    debug_print("evaluate_model", "Starting evaluation on the test set.")

    model.eval()
    criterion = nn.L1Loss() #nn.MSELoss()
    test_loader = tg.loader.DataLoader(dataset.test_dataset, batch_size=1, shuffle=False)

    test_loss = 0.0
    preds = []
    truths = []

    with torch.no_grad():
        for batch in test_loader:
            if not hasattr(batch, 'batch'):
                batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)

            batch = batch.to(device)
            y_pred = model(batch)
            loss = criterion(y_pred, batch.y)
            test_loss += loss.item()

            preds.append(y_pred.item())
            truths.append(batch.y.item())

    if len(test_loader) > 0:
        test_loss /= len(test_loader)

    print(f"MAE on test set: {test_loss:.4f}")
    return preds, truths, test_loss

import matplotlib.pyplot as plt

def plot_metrics(train_losses, test_losses):
    """
    Plots the train and test loss curves along the epochs.
    """
    print(len(train_losses))
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(train_losses, label="Train Loss", color='blue', linewidth=2, linestyle='-', marker='o')
    plt.plot(test_losses, label="Test Loss", color='red', linewidth=2, linestyle='-', marker='x')

    plt.title("Train and Test Loss over Epochs", fontsize=16)  # Title of the plot
    plt.xlabel('Epochs', fontsize=14)  # X-axis label
    plt.ylabel('Loss (MAE)', fontsize=14)  # Y-axis label
    plt.legend(loc='upper right', fontsize=12)  # Display legend

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def visualize_layers(model):
    print("==== Model Visualization ====")
    print("Model type:", type(model).__name__)

    print("\nEmbedding layer:")
    print(" ", model.embedding)

    print("\nConvolution layers (message-passing):")
    for i, conv in enumerate(model.conv_layers):
        print(f"  ConvLayer {i} : {type(conv).__name__}")
        print("    node_mlp:", conv.node_mlp)
        print("    edge_mlp:", conv.edge_mlp)

        node_params = sum(p.numel() for p in conv.node_mlp.parameters())
        edge_params = sum(p.numel() for p in conv.edge_mlp.parameters())
        print(f"    # node_mlp parameters: {node_params}")
        print(f"    # edge_mlp parameters: {edge_params}")

    print("\nReadout MLP:")
    print(model.readout_mlp)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameter count in model: {total_params}\n")


def plot_predictions(model, dataset, device='cpu'):
    """
    Plots the predicted values with a 95% confidence interval and the actual values on the test set.

    Parameters:
        model: The trained model.
        dataset: The dataset object with the test data.
        device: The device ('cpu' or 'cuda') where the model is located.
    """
    model.eval()  # Set model to evaluation mode

    # Prepare data loader for test set
    test_loader = DataLoader(dataset.test_dataset, batch_size=1, shuffle=False)

    # Lists to hold true values, predictions, and prediction errors
    y_true = []
    y_pred = []
    prediction_errors = []

    with torch.no_grad():  # No gradient calculation for inference
        for batch in test_loader:
            # Move batch to the specified device
            batch = batch.to(device)

            # Make predictions
            pred = model(batch).cpu().numpy()  # Convert predictions to NumPy array
            true = batch.y.cpu().numpy()  # Actual values

            # Collect true values and predictions
            y_true.append(true)
            y_pred.append(pred)
            prediction_errors.append(true - pred)

    # Convert lists to numpy arrays for easier handling
    y_true = np.array(y_true).flatten()  # Flatten to 1D
    y_pred = np.array(y_pred).flatten()
    prediction_errors = np.array(prediction_errors).flatten()

    # Calculate 95% confidence intervals for the predictions
    # Use the standard error of the prediction errors
    mean_error = np.mean(prediction_errors)
    std_error = np.std(prediction_errors)
    n = len(prediction_errors)

    # Standard error of the mean (SEM)
    sem = std_error / np.sqrt(n)

    # Plot the actual vs predicted values with the confidence interval
    plt.figure(figsize=(10, 6))

    # Plot predicted vs actual values
    plt.plot(y_true, y_pred, 'bo', label='Predictions', markersize=5)

    # Plot perfect prediction line (y = x)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction (y=x)')

    # Labels and title
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title('Predictions vs Actual with 95% Confidence Interval', fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_print("main", "Main script started.")

    dataset = MolecularDataset('test_dataset.json', test_size=0.2)

    model = train_model(dataset, num_epochs=50, batch_size=32, device='cpu', patience=10)

    preds, truths, mae_test = evaluate_model(model, dataset, device='cpu')

    type_info = {
        'type_encoding': dataset.type_encoding,
        'type_onehot': dataset.type_onehot.tolist()
    }
    with open('type_info.json', 'w') as f:
        json.dump(type_info, f)
    visualize_layers(model)
    # Plot metrics at the end
    debug_print("main", "Main script finished.")

    # Example usage
    # Assuming 'model' is your trained model, and 'dataset' is your dataset with a 'test_dataset' property
    plot_predictions(model, dataset, device='cpu')
