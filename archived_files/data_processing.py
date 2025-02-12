import re
import torch
import numpy as np
from e3nn import o3
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler
from ase import Atoms
from ase.data import atomic_numbers
from torch_cluster import radius_graph
from config import DEVICE, CUTOFF_RADIUS

class AtomDataProcessor:
    def __init__(self, data_list):
        self.data_list = data_list

    def initialize_atom_data(self):
        """Initializes atomic symbol and position data based on the list of dictionaries."""
        atom_pattern = r'([A-Za-z]+)\s([-+]?\d*\.\d+|\d+)\s([-+]?\d*\.\d+|\d+)\s([-+]?\d*\.\d+)'  # Regex for atom data

        for mol in self.data_list:
            # Extract atom data for each molecule
            mol_coordinates = mol["optimized_coordinates_S0"]
            symbols = []
            positions = []
            for atom in mol_coordinates[1:]:  # Start from the second element as the first is the molecule identifier
                match = re.match(atom_pattern, atom)
                if match:
                    symbol = match.group(1)
                    x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
                    symbols.append(symbol)
                    positions.append([x, y, z])

            # Add the extracted atom features to the existing molecule data
            mol['symbols'] = symbols
            mol['positions'] = positions

        return self.data_list  # Return updated list of molecules with added atom features


class MolecularDataset(Dataset):
    def __init__(self, data_list, scaler=None):
        super().__init__()

        self.data_list = data_list
        self.scaler = scaler or self._initialize_scalers()

        # Initialize atom data using AtomDataProcessor
        processor = AtomDataProcessor(data_list)
        self.data_list = processor.initialize_atom_data()

        self._validate_data()

    def _initialize_scalers(self):
        """Initializes feature scalers based on the dataset."""
        features = {'homo': [], 'lumo': [], 'dipole': [], 'reduction_potential': []}
        for mol in self.data_list:
            features['homo'].append(float(mol.get('homo (eV)', 0.0)))
            features['lumo'].append(float(mol.get('lumo (eV)', 0.0)))
            features['dipole'].append(mol.get('dipole_moment_vector_S1 (D)', [0.0, 0.0, 0.0]))
            features['reduction_potential'].append(mol.get('reduction_potential_S1 (eV)'))

        return {
            'homo': StandardScaler().fit(np.array(features['homo']).reshape(-1, 1)),
            'lumo': StandardScaler().fit(np.array(features['lumo']).reshape(-1, 1)),
            'dipole': StandardScaler().fit(np.array(features['dipole']))
        }

    def _validate_data(self):
        """Filters invalid molecules from the dataset."""
        valid_indices = [i for i in range(len(self.data_list)) if self.get(i) is not None]
        self.data_list = [self.data_list[i] for i in valid_indices]
        if not self.data_list:
            raise ValueError("No valid molecules found in dataset!")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """Converts raw molecular data into a PyG Data object."""
        try:
            mol = self.data_list[idx]
            target = torch.tensor([float(mol.get('reduction_potential_S1 (eV)', 0.0))], dtype=torch.float32).to(DEVICE)
            atoms = Atoms(symbols=mol['symbols'], positions=mol['positions'])
            pos = torch.tensor(atoms.positions, dtype=torch.float32).to(DEVICE)
            edge_index = radius_graph(pos, r=CUTOFF_RADIUS)
            node_z = torch.tensor([atomic_numbers[s] for s in mol['symbols']], dtype=torch.long).unsqueeze(-1).to(
                DEVICE)
            return Data(x=node_z, pos=pos, edge_index=edge_index, y=target)
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            return None

    def print_sample(self, n=3):
        """Prints a sample of `n` molecules from the dataset."""
        n = min(n, len(self.data_list))  # Ensure we don’t go out of bounds
        print(f"Printing {n} sample molecules from the dataset:\n")

        for i in range(n):
            mol = self.data_list[i]
            print(f"Molecule {i + 1}:")
            print(f" Reduction Potential (S1): {mol['reduction_potential_S1 (eV)']}")
            print(f"  Symbols: {mol['symbols']}")
            print("  Positions:")
            for symbol, pos in zip(mol['symbols'], mol['positions']):
                print(f"    {symbol}: {pos}")
            print("\n" + "-" * 40 + "\n")  # Separator for readability
