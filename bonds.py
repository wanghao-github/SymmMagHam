import numpy as np
from pymatgen.core import Structure, Lattice

class Bond:
    def __init__(self, supercell):
        self.supercell = supercell
        self.atom_positions = [atom.coords for atom in supercell]
        self.atom_indices = list(range(len(supercell)))
        self.distance_matrix = supercell.distance_matrix

    def __repr__(self):
        return f"Bond(Class with {len(self.supercell)} atoms)"