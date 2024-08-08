import numpy as np
import ase 
import spglib as spg
import pymatgen
from math import *
import csv
from pymatgen.io.cif import CifParser
from spglib import get_spacegroup
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core import Lattice, Structure
from copy import deepcopy

def mod_one(arr):
    return np.mod(arr, 1)

parser = CifParser(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\space_group\P3_cell.cif')

structure = parser.parse_structures()[0]

# 获取晶胞参数和原子位置
lattice = structure.lattice.matrix
positions = structure.frac_coords
atomic_numbers = [site.specie.number for site in structure]

# print(atomic_numbers)

cell_nomag = (lattice, positions, atomic_numbers)

symmetrized_cell_nomag = spg.standardize_cell(cell_nomag, to_primitive=False, no_idealize=False, symprec=1e-4)


print(symmetrized_cell_nomag)
symmetry_nomag = spg.get_symmetry_dataset(symmetrized_cell_nomag,symprec=1e-3)

# print(symmetry_nomag)

std_positions =symmetry_nomag['std_positions']
std_lattice =symmetry_nomag['std_lattice']

species = ["Cr","Cr","Cr"]
primitive_cell = Structure(std_lattice, species,std_positions)

structure = deepcopy(primitive_cell)

radius = 10

scaling_matrix = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
supercell = structure.make_supercell(np.array(scaling_matrix).diagonal())

super_lattice = supercell.lattice.matrix
super_positions = supercell.frac_coords

# super_species = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
# super_species = ["Cr","Cr","Cr","Cr","Cr","Cr","Cr","Cr","Cr","Cr","Cr","Cr"]
super_species = [6, 6, 6]
scell = (super_lattice,super_positions,super_species)


print(scell)
symm_data=spg.get_symmetry_dataset(scell,symprec=1e-5)


print(symm_data)
print(symm_data['rotations'])
print(symm_data['translations'])

tolerance = 1e-4

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

def are_parallel_or_antiparallel(vec1, vec2, tolerance=1e-4):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(vec1, vec2)
    if abs(dot_product - 1) < tolerance:
        return True
    elif abs(dot_product + 1) < tolerance:
        return True
    else:
        return False

def is_lattice_translation(vec_diff, lattice_vectors, tolerance=1e-4):
    try:
        coeffs = np.linalg.solve(lattice_vectors, vec_diff)
        if np.allclose(coeffs, np.round(coeffs), atol=tolerance):
            return True
    except np.linalg.LinAlgError:
        return False
    return False

def rounded_length(length, tolerance=0.001):
    return round(length / tolerance) * tolerance


bonds = []
for i, site in enumerate(supercell):
    neighbors = supercell.get_neighbors(site, r=radius)
    for neighbor in neighbors:
        j = neighbor.index
        if j >= i:
            img = neighbor.image
            end_coords = supercell.lattice.get_cartesian_coords(neighbor.frac_coords)
            vector = end_coords - site.coords
            length = np.linalg.norm(vector)
            rounded_len = rounded_length(length)
            bonds.append((i, j, img, rounded_len, site.coords, end_coords))
print(bonds)

length_groups = {}
for idx, bond in enumerate(bonds):
    _, _, _, length, _, _ = bond
    if length not in length_groups:
        length_groups[length] = []
    length_groups[length].append((idx, bond))
print(length_groups[10.0])

operation_to_bonds = {}
for length, group in length_groups.items():
    if length not in operation_to_bonds:
        operation_to_bonds[length] = {}
    for idx1, bond1 in group:
        start1, end1, img1, _, start_coords1, end_coords1 = bond1
        print(bond1)
        vec1= end_coords1-start_coords1
        for idx2, bond2 in group:
            start2, end2, img2, _, start_coords2, end_coords2 = bond2
            vec2 = end_coords2 - start_coords2
            if idx1 < idx2: 
                matching_ops = []
                for op in range(len(symm_data['rotations'])):
                    rot = symm_data['rotations'][op]
                    trans = symm_data['translations'][op]
                        # a = supercell.lattice.get_cartesian_coords(trans)
                        # b =  supercell.lattice.get_cartesian_coords(img1)
                    transformed_vec1  = np.dot(rot, vec1) + supercell.lattice.get_cartesian_coords(trans) #+ supercell.lattice.get_cartesian_coords(img1)
                    if are_parallel_or_antiparallel(transformed_vec1, vec2) and is_lattice_translation(transformed_vec1 - vec2, supercell.lattice.matrix):
                        matching_ops.append(op)
                # print(idx1,idx2)
                # print(matching_ops)    
                if matching_ops:
                    op_key = frozenset(matching_ops)  # 使用操作集合作为键
                    if op_key not in operation_to_bonds[length]:
                        operation_to_bonds[length][op_key] = []
                    operation_to_bonds[length][op_key].append((idx1, idx2))
# 分析每组键共有的对称操作
for length, op_groups in operation_to_bonds.items():
    print(f"length {length}: {len(op_groups)} groups of symmetry-related bonds:")
    for group_idx, (ops, bonds) in enumerate(op_groups.items(), 1):
        print(f"  Group {group_idx}: operations {list(ops)}")
        for bond1, bond2 in bonds:
            print(f"    Bond {bond1},{bond2}")