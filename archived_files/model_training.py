import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from archived_files.config import CONFIG, DEVICE
from model import ReductionPotentialPredictor
from archived_files.data_processing import MolecularDataset
from utils import debug_print, plot_loss_curves
import json
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self._init_data()
        self._init_model()

    def _init_data(self):
        with open(self.config['data_path']) as f:
            full_data = json.load(f)
        train_data, val_data = train_test_split(full_data, test_size=self.config['test_size'],
                                                random_state=self.config['seed'])
        self.train_set = MolecularDataset(train_data)
        self.val_set = MolecularDataset(val_data)
        self.train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config['batch_size'])

    def _init_model(self):
        self.model = ReductionPotentialPredictor().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])
        self.loss_fn = nn.HuberLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = batch.to(DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.train_set)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(DEVICE)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.val_set)

    def run_training(self):
        best_loss = float('inf')
        train_losses, val_losses = [], []
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            plot_loss_curves(train_losses, val_losses, epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.config['save_path'])
        debug_print("Training complete", data={'best_val_loss': best_loss})


if __name__ == "__main__":
    trainer = ModelTrainer(CONFIG)
    trainer.run_training()
