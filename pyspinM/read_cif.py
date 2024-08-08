import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
import spglib
from copy import deepcopy
from  math import ceil
from numpy.linalg import inv, norm


def parse_and_symmetrize_structure(cif_path):
    
    parser = CifParser(cif_path)
    cif_structure = parser.parse_structures()[0]

    lattice = cif_structure.lattice.matrix
    positions = cif_structure.frac_coords
    atomic_numbers = [site.specie.number for site in cif_structure]

    cell = (lattice, positions, atomic_numbers)
    
    symmetrized_cell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=1e-4)
    symmetry_dataset = spglib.get_symmetry_dataset(symmetrized_cell, symprec=1e-3)

    std_positions = symmetry_dataset['std_positions']
    std_lattice = symmetry_dataset['std_lattice']
    std_types = symmetry_dataset['std_types']

    atom_species = [Element.from_Z(z).symbol for z in std_types]
    symmetrized_structure = Structure(std_lattice, atom_species, std_positions)
    
    return symmetry_dataset, symmetrized_structure

# cif_file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\VO2P-4m2.cif'
# cif_file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\VO2P-4m2.cif'
cif_file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\P3_cell.cif'
# cif_file_path2 = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\CrI3.cif'
symmetry_dataset, result_structure = parse_and_symmetrize_structure(cif_file_path)
# symmetry_dataset2, result_structure2 = parse_and_symmetrize_structure(cif_file_path2)
# print(symmetry_dataset2)
# print(result_structure2)

structure = deepcopy(result_structure)
radius = 6

def find_and_store_bonds(structure, radius):
    bonds = []
    dr_tracker = {}
    total_neighbors = 0

    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=radius)
        total_neighbors += len(neighbors)

        for neighbor in neighbors:
            idx1 = i
            idx2 = neighbor.index
            dr = neighbor.frac_coords - site.frac_coords
            dl = np.array(neighbor.image)
            length = neighbor.nn_distance
            length_rounded = round(length, 3)
            add_bond = False
            bond_info = {
                'idx': None,
                'subidx': len(bonds),
                'dl': dl,
                'dr': dr,
                'length': length_rounded,
                'matom1': site.specie.symbol,
                'matom2': neighbor.specie.symbol,
                'idx1': idx1,
                'idx2': idx2,
                'matrix': {}
            }
            if idx1 == idx2:
                if idx1 in dr_tracker and any(np.allclose(dr, -np.array(x), atol=1e-5) for x in dr_tracker[idx1]):
                    continue
                dr_tracker.setdefault(idx1, []).append(dr)
                add_bond = True
            elif idx2 > idx1:
                add_bond = True
            if add_bond:
                bonds.append(bond_info)

    return bonds, total_neighbors

def verify_bonds_count(structure, bonds, total_neighbors):
    return len(bonds) * 2 == total_neighbors

def assign_idx_and_sort_bonds(bonds):
    sorted_bonds = sorted(bonds, key=lambda x: x['length'])
    current_idx = 0
    current_length = None
    for bond in sorted_bonds:
        if bond['length'] != current_length:
            current_length = bond['length']
            current_idx += 1
        bond['idx'] = current_idx
    return sorted_bonds

bonds_list, total_neighbors = find_and_store_bonds(structure, radius)
verification_result = verify_bonds_count(structure, bonds_list, total_neighbors)
sorted_bonds = assign_idx_and_sort_bonds(bonds_list)

def is_same_bond(dr1, dr2, idx1, idx2, tran_idx1, tran_idx2, tol=1e-4):
    if  (np.allclose(dr1, -dr2, atol=tol) and idx1 == tran_idx2 and idx2 == tran_idx1) or \
        (np.allclose(dr1, dr2, atol=tol) and idx1 == tran_idx1 and idx2 == tran_idx2):
        return True
    else:
        return False

def atom_symmetry_mapping(positions, rot_ops, trans_ops):
    op_mapping_table = {}
    for i, rot in enumerate(rot_ops):
        trans = trans_ops[i]
        atom_map = {}
        for j, pos in enumerate(positions):
            original_pos = pos
            new_pos = np.dot(rot, original_pos) + trans
            new_pos = new_pos % 1
            distances = np.linalg.norm(positions - new_pos , axis=1)
            closest_atom_index = np.argmin(distances)
            atom_map[j] = closest_atom_index
        op_mapping_table[i] = atom_map
    return op_mapping_table

sym_map_dict=atom_symmetry_mapping(symmetry_dataset['std_positions'],symmetry_dataset['rotations'],symmetry_dataset['translations'])
print(sym_map_dict)
def classify_bonds_by_symmetry(bonds, sym_map_dict, symmetry_dataset):
    grouped_bonds_by_length = {}
    for bond in bonds:
        idx = bond['idx']
        if idx not in grouped_bonds_by_length:
            grouped_bonds_by_length[idx] = []
        grouped_bonds_by_length[idx].append(bond)

    final_groups = {}

    for idx, bonds_with_same_len in grouped_bonds_by_length.items():
        groups = []
        for subidx,each_bonds in enumerate(bonds_with_same_len):
            found_group = False
            current_bond = each_bonds
            # print("current_bond",current_bond)
            # print(bonds_with_same_len)
            current_dr = np.array(current_bond['dr'])
            current_idx1 = current_bond['idx1']
            current_idx2 = current_bond['idx2']
            current_dl   = current_bond['dl']
            current_length = current_bond['length']
            current_matom1 = current_bond['matom1']
            current_matom2 = current_bond['matom2']
            print("current idx1, idx2, dr is ",current_dr,current_idx1,current_idx2)
            transformed_current_bonds = []
                     
            for group in groups:
                match_found = False
                for j, (rot, trans) in enumerate(zip(symmetry_dataset['rotations'], symmetry_dataset['translations'])):
                    transformed_dr = np.dot(rot, current_dr) + trans
                    transformed_idx1 = sym_map_dict[j][current_idx1]
                    transformed_idx2 = sym_map_dict[j][current_idx2]
                    transformed_current_bonds.append((transformed_idx1, transformed_idx2, transformed_dr))
                    print("transformed_current_bonds",transformed_current_bonds)
                    for other_idx1, other_idx2, other_dr,*args in group:
                        print("other_idx1, other_idx2, other_dr are",other_idx1, other_idx2, other_dr)
                        if is_same_bond(other_dr+trans, transformed_dr, other_idx1, other_idx2, transformed_idx1, transformed_idx2):
                            match_found = True
                            break
                    if match_found:
                        break
                if match_found:
                    group.append((current_idx1, current_idx2, current_dr,current_dl,current_length,current_matom1,current_matom2))
                    found_group = True
                    break
            if not found_group:
                groups.append([(current_idx1, current_idx2, current_dr,current_dl,current_length,current_matom1,current_matom2)])
            print(groups)
        final_groups[idx] = groups 

    return final_groups

bonds_list, total_neighbors = find_and_store_bonds(structure, radius)
sorted_bonds = assign_idx_and_sort_bonds(bonds_list)
classified_groups = classify_bonds_by_symmetry(sorted_bonds, sym_map_dict,symmetry_dataset)

def print_grouped_bonds(final_groups):
    for length_idx, groups in final_groups.items():
        print(f"键长索引 {length_idx} 下的分组：")
        for idx, group in enumerate(groups, start=1):
            print(f"  分组 {idx} 包括如下键的原始和变换后的 dr 向量：")
            for bond in group:
                print(f"    原始索引: {bond[0]}-{bond[1]}, dr 向量: {bond[2]}")

# classified_groups = classify_bonds_by_symmetry(bonds, sym_map_dict, rotations, translations)
print_grouped_bonds(classified_groups)

def reset_idx_subidx_dict(original_dict):
    new_dict = {}  # 新词典，存放重组后的数据
    new_key_counter = 1  # 新词典的键的计数器
    # 遍历原始词典中的每个键和对应的值（列表的列表）
    for key, list_of_lists in original_dict.items():
        for sublist in list_of_lists:
            new_dict[new_key_counter] = {}  # 为每个列表创建一个嵌套词典     
            subIdx = 1  # 初始化subIdx，对子列表中的每个元组进行编号
            for tup in sublist:
                # 将每个元组存储在其对应的subIdx下
                new_dict[new_key_counter][subIdx] = tup
                subIdx += 1  # 更新subIdx
            new_key_counter += 1  # 更新外层字典的键
    return new_dict

bonds_dict = reset_idx_subidx_dict(classified_groups)
print(bonds_dict)

def center_of_bond(structure,bonds_dict):
    center_of_each_symbonds = {}
    for idx,symbonds in bonds_dict.items():
        # 
        center_of_each_symbonds[idx] ={}
        for subidx,each_bonds in symbonds.items():
            start_pos = structure.frac_coords[each_bonds[0]]
            end_pos = start_pos + each_bonds[2]
            center = ((start_pos + end_pos) / 2) % 1 
            center_of_each_symbonds[idx][subidx] = center
            # print(center)
        # start_pos = structure.frac_coords[symbonds[1][0]]    
        # end_pos = start_pos + symbonds[1][2]
        # center = ((start_pos + end_pos) / 2) % 1
        # center_of_each_symbonds[idx] = center
    return center_of_each_symbonds
    # return pass
print(structure.frac_coords)
# print(sorted_bonds)
center_pos = center_of_bond(structure,bonds_dict)
print(center_pos[4][1])

