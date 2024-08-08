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

parser = CifParser(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\CrI3_22supercell.cif')
# parser = CifParser(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\EuMnBi2.cif')
# print("Hello World")
# print(parser)
structure = parser.parse_structures()[0]

# 获取晶胞参数和原子位置
lattice = structure.lattice.matrix
positions = structure.frac_coords
# print(positions)
atomic_numbers = [site.specie.number for site in structure]
# magmoms = [[0,0,5],[0,0,-5],[0,0,5],[0,0,-5],[0,0,5],[0,0,-5],[0,0,5],[0,0,-5],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0]]
# magmoms = [[0,0,5],[0,0,5],[0,0,5],[0,0,5],[0,0,-5],[0,0,-5],[0,0,-5],[0,0,-5],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0]]
magmoms = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0]]

# 构建spglib所需的晶体结构
# magmoms = [[5,0,0],[5,0,0],[5,0,0],[5,0,0],[-5,0,0],[-5,0,0],[-5,0,0],[-5,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0]]

# magmoms =  [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[5,0,0],[5,0,0],[5,0,0],[5,0,0],[0,0,0], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0], [0,0,0]]
cell = (lattice, positions, atomic_numbers,magmoms)

cell_nomag = (lattice, positions, atomic_numbers)
symmetrized_cell_nomag = spg.standardize_cell(cell_nomag, to_primitive=False, no_idealize=False, symprec=1e-3)

print(cell)

symmetry_nomag = spg.get_symmetry_dataset(symmetrized_cell_nomag,symprec=1e-3)
# print(symmetry_nomag)

# space_group = SpaceGroup.from_int_number(194)

# print(f"Crystal system: {space_group.crystal_system}")
# print(f"Point group: {space_group.point_group}")

# new_lattice, new_positions, new_elements = symmetrized_cell
# new_mag_cell = (new_lattice, new_positions, new_elements, magmoms)

# print("cell is,", new_mag_cell)
# print("symmetrized_cell is", symmetrized_cell)
# # # 使用spglib获取空间群信息
# spacegroup = get_spacegroup(symmetrized_cell, symprec=1e-2)
# print(spacegroup)
symmetry = spg.get_symmetry_dataset(cell,symprec=1e-3)
mag_dataset = spg.get_magnetic_symmetry_dataset(cell,symprec=1e-3)

# print(type(mag_dataset['uni_number']))
print(symmetry)
print(mag_dataset)

def build_prototype(space_group_number = None):
    
    if space_group_number >= 1 and space_group_number <= 2:
        print("this structure is Triclinic")
        a, b, c = 5.2, 7.1, 5.3  # 晶格常数，单位为Å
        alpha, beta, gamma = 91, 109, 92  # 晶格角度，单位为度
        # 创建三斜晶系晶格
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    if space_group_number >= 3 and space_group_number <= 15:
        print("this structure is Monoclinic")
        a, b, c = 5.679, 15.202, 6.522  # 晶格常数，单位为Å
        beta = 118.43  # β角，单位为度
        # 创建单斜晶系晶格
        lattice = Lattice.monoclinic(a, b, c, beta)
        
    if space_group_number >= 16 and space_group_number <= 74:
        print("this structure is Orthorhombic")
        a, b, c = 4.96, 7.97, 5.74  # 晶格常数，单位为Å
        alpha, beta, gamma = 90, 90, 90  # 所有角度均为90°
        # 创建正交晶系晶格
        lattice = Lattice.orthorhombic(a, b, c)
        
    if space_group_number >= 75 and space_group_number <= 142:
        print("this structure is Tetragonal")
        lattice = Lattice.tetragonal(4.737, 3.186)
        
    if space_group_number >= 143 and space_group_number <= 167:
        print("this structure is Trigonal")
        lattice = Lattice.hexagonal(2.46, 6.70)
        
    if space_group_number >= 168 and space_group_number <= 194:
        print("this structure is Hexagonal")
        lattice = Lattice.hexagonal(2.46, 6.70)
        
    if space_group_number >= 195 and space_group_number <= 230:
        print("this structure is Cubic")
        lattice = Lattice.cubic(5.64)
    return lattice

# print(build_prototype(111))

# def construct_mag_prim_cell():
# print(symmetry['number'])
# print("旋转矩阵:")
# for i, rot in enumerate(symmetry['rotations']):
#     print(f"操作 {i+1}:\n", rot)

# print("\n平移向量:")
# for i, trans in enumerate(symmetry['translations']):
#     print(f"操作 {i+1}:", trans)
    
# symm =  spg.get_magnetic_symmetry_dataset(cell,symprec=1e-2)
# print(symm)

def get_msg_numbers():
    all_datum = []
    with open(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\msg_numbers.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            if len(row) == 0:
                break

            litvin_number, bns_number, og_number, uni_number = row
            all_datum.append((
                int(litvin_number),
                bns_number,
                og_number,
                int(uni_number),
            ))

    assert len(all_datum) == 1651
    return all_datum

# aa  = get_msg_numbers()[1650][2].strip().split('.')[-1]
# print(aa)

