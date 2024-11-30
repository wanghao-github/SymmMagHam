import numpy as np
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core import Structure, Element,Lattice
import spglib
from copy import deepcopy
from math import ceil
from numpy.linalg import inv, norm
from scipy.linalg import null_space, orth
from itertools import combinations
import os
import shutil
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
# import numpy as np
# import os
# import shutil
def print_matrix(matrix):
    for i in range(matrix.shape[2]):
        print(matrix[:,:,i])

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

def find_and_store_bonds(structure, radius):
    bonds = []
    dr_tracker = {}
    total_neighbors = 0
    magnetic_elements = {'Cu','Fe','Co', 'Cr', 'Mn','O', 'V', 'Ni'}
    image_x = []
    image_y = []
    image_z = []
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=radius)
        total_neighbors += len(neighbors)
        # print(neighbors)
        for neighbor in neighbors:
            idx1 = i
            idx2 = neighbor.index
            dr = neighbor.frac_coords - site.frac_coords
            dl = np.array(neighbor.image)
            image_x.append(dl[0])
            image_y.append(dl[1])
            image_z.append(dl[2])
            length = neighbor.nn_distance
            length_rounded = round(length, 4)
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
            
            if site.specie.symbol in magnetic_elements and neighbor.specie.symbol in magnetic_elements:
                if idx1 == idx2:
                    if idx1 in dr_tracker and any(np.allclose(dr, -np.array(x), atol=1e-3) for x in dr_tracker[idx1]):
                        continue
                    dr_tracker.setdefault(idx1, []).append(dr)
                    add_bond = True
                elif idx2 > idx1:
                    add_bond = True
                
            if add_bond:
                bonds.append(bond_info)
    image_differences = (max(image_x) - min(image_x), max(image_y) - min(image_y), max(image_z) - min(image_z))
    return bonds, total_neighbors,image_differences

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

def is_same_bond(dr1, dr2, idx1, idx2, tran_idx1, tran_idx2, tol=1e-3):
    if  (np.allclose(dr1, -dr2, atol=tol) and idx1 == tran_idx2 and idx2 == tran_idx1) or \
        (np.allclose(dr1, dr2, atol=tol) and idx1 == tran_idx1 and idx2 == tran_idx2):
        # (np.allclose(dr1, -dr2, atol=tol) and idx1 == tran_idx1 and idx2 == tran_idx2):
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
        # transformed_current_bonds = []
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
                # print("group is ",group)
                match_found = False
                for j, (rot, trans) in enumerate(zip(symmetry_dataset['rotations'], symmetry_dataset['translations'])):
                    transformed_dr = np.dot(rot, current_dr)
                    transformed_idx1 = sym_map_dict[j][current_idx1]
                    transformed_idx2 = sym_map_dict[j][current_idx2]
                    transformed_current_bonds.append((transformed_idx1, transformed_idx2, transformed_dr))
                    # print("j is",j,"trans is",trans)
                    for other_idx1, other_idx2, other_dr,*args in group:
                        if is_same_bond(other_dr, transformed_dr, other_idx1, other_idx2, transformed_idx1, transformed_idx2):
                            # print("the same bond with ",other_dr+trans, transformed_dr, other_idx1, other_idx2, transformed_idx1, transformed_idx2)
                            match_found = True
                            break
                    if match_found:
                        break
                if match_found:
                    # print("this bond already existed")
                    group.append((current_idx1, current_idx2, current_dr,current_dl,current_length,current_matom1,current_matom2,j))
                    found_group = True
                    break
            if not found_group:
                # print("add this bond!")
                groups.append([(current_idx1, current_idx2, current_dr,current_dl,current_length,current_matom1,current_matom2,0)])
        final_groups[idx] = groups 
    return final_groups

def print_grouped_bonds(final_groups):
    for length_idx, groups in final_groups.items():
        print(f"键长索引 {length_idx} 下的分组：")
        for idx, group in enumerate(groups, start=1):
            print(f"  分组 {idx} 包括如下键的原始和变换后的 dr 向量：")
            for bond in group:
                print(f"    键长:{bond[4]} 原始索引: {bond[0]}-{bond[1]}, dr 向量: {bond[2]}, dl 向量: {bond[3]}, 对称性操作:{bond[7]} ")

def reset_idx_subidx_dict(original_dict):
    new_dict = {}  # 新词典，存放重组后的数据
    new_key_counter = 1  # 新词典的键的计数器
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

def center_of_bond(structure,bonds_dict):
    center_of_each_symbonds = {}
    for idx,symbonds in bonds_dict.items():
        # 
        center_of_each_symbonds[idx] ={}
        for subidx,each_bonds in symbonds.items():
            start_pos = structure.frac_coords[each_bonds[0]]
            end_pos = start_pos + each_bonds[2]
            center = ((start_pos + end_pos) / 2) % 1 
            center_of_each_symbonds[idx][subidx] = np.array(center)

    return center_of_each_symbonds

def intersec(U1, U2):
    N = null_space(np.hstack([U1, U2]))
    I = U1 @ N[:U1.shape[1], :]
    return orth(I)

def indep(M, v, tol):

    rank_M = np.linalg.matrix_rank(M, tol=tol)
    independent_indices = []
    for idx in range(v.shape[1]):
        augmented_matrix = np.hstack([M, v[:, [idx]]])
        rank_augmented = np.linalg.matrix_rank(augmented_matrix, tol=tol)
        if rank_augmented > rank_M:
            independent_indices.append(idx)
    return v[:, independent_indices]

def oporder(symOp):
    if symOp.size == 0:
        print("Help on function 'oporder' in module 'swsym':")
        print("oporder(symOp)")
        print("    Calculate the order of a symmetry operation represented by symOp.")
        return None
    N = 1
    RN = symOp[:, 0:3]
    TN = np.round(symOp[:, 3] * 12)
    eps = np.finfo(float).eps

    while (np.linalg.norm(RN - np.eye(3)) > 10 * eps or np.linalg.norm(TN)) and (N < 10):
        RN = RN @ symOp[:, 0:3]
        TN = np.mod(np.round(symOp[:, 0:3] @ TN + symOp[:, 3]), 12)
        N += 1

    return N

def basic_sym_matrix(symOp,r,tol):
    nSym = symOp.shape[2]
    V0 = np.zeros((9, 9))
    # 对称部分
    V0[0, 0] = 1; V0[4, 1] = 1; V0[8, 2] = 1
    V0[[1, 3], 3] = 1
    V0[[2, 6], 4] = 1
    V0[[5, 7], 5] = 1
    # 反对称部分
    V0[[5, 7], 6] = [-1, 1]
    V0[[2, 6], 7] = [1, -1]
    V0[[1, 3], 8] = [-1, 1]

    M0_V = V0

    # print("M0_V",M0_V)
    norm_r = np.linalg.norm(r)
    if norm_r > 0:
        r = r / norm_r
        aniso = False
    else:
        print("aniso = True !!")
        aniso = True

    for ii in range(nSym):
        
        R = symOp[:, :, ii]
        # R= symOp[ii,:,:]
        if not aniso:
            ordR = 1
            while np.abs(np.abs(r.T @ np.linalg.matrix_power(R, ordR) @ r) - 1) > tol and ordR < 10:
                ordR += 1

            if ordR == 10:
                raise Exception('Not a valid point group generator!')

            R = np.linalg.matrix_power(R, ordR)
            parR = np.sign(r.T @ R @ r)

        else:
            ordR = oporder(np.hstack([R, np.zeros((3, 1))]))
            parR = 1
            
        # print("ordR is",ordR)
        # print("parR is",parR)
        I9 = np.eye(9)
        kron_R_R = np.kron(R, R)
        U, d, MS = np.linalg.svd(kron_R_R - I9)

        D = np.zeros((U.shape[1], MS.shape[0]), dtype=np.float64)
        np.fill_diagonal(D, d)   
        M_S = MS.T[:, np.abs(np.diag(D)) < tol].reshape(3, 3, -1,order='F')
        if parR == -1:
            U, d, MA = np.linalg.svd(kron_R_R + I9)
            D = np.zeros((U.shape[1], MA.shape[0]), dtype=np.float64)
            np.fill_diagonal(D, d)
            # print("MA is\n",MA.T)
            M_A = MA.T[:,np.abs(np.diag(D)) < tol].reshape(3, 3, -1, order='F')
            M_A = M_A - np.transpose(M_A, (1, 0, 2))
            
        else:
            M_A = M_S - np.transpose(M_S, (1, 0, 2))

        M_S = M_S + np.transpose(M_S, (1, 0, 2))
        M_V = np.reshape(np.concatenate((M_S, M_A), axis=2), (9, -1 ))

        # print("M0_V is",M0_V)
        M0_V = intersec(M0_V, M_V)

        normM = np.array([np.linalg.norm(M0_V[:, idx]) for idx in range(M0_V.shape[1])])
        # print(normM)
        M0_V = M0_V[:, normM >= tol]
        # print(M0_V)
    
    M0_V = M0_V.reshape(3, 3, -1, order='F')
    # M0_V = aaa.reshape(3, 3, -1,order='F')
    # print_matrix(M0_V)
    symmetric = M0_V + np.transpose(M0_V, (1, 0, 2))  # 计算对称部分
    antisymmetric = M0_V - np.transpose(M0_V, (1, 0, 2))  # 计算反对称部分
    M0_V = np.concatenate((antisymmetric, symmetric), axis=2).reshape(9, -1)  # 重塑回 (9, num_matrices*2)
    
    normM = np.linalg.norm(M0_V, axis=0)
    M0_V = M0_V[:, normM >= tol]

    rank_M0_V = np.linalg.matrix_rank(M0_V, tol=tol)
    # print(rank_M0_V)
    rM = np.array([np.linalg.matrix_rank(np.hstack([M0_V, V0[:, [idx]]]), tol=tol) for idx in range(V0.shape[1])])

    Vnice = V0[:, rM == rank_M0_V]
    # print("Vnice is", Vnice)
    for ii in range(M0_V.shape[1]):
        column_vector = M0_V[:, ii].reshape(-1, 1)
        addV = indep(Vnice, column_vector, tol)
        if addV.size > 0:
            addV = addV - np.sum(np.sum(addV * Vnice, axis=0) * Vnice, axis=1).reshape(-1, 1)
            Vnice = np.hstack([Vnice, addV])
    
    M = Vnice
    divM = np.array([M[np.argmax(np.abs(M[:, idx])), idx] for idx in range(M.shape[1])])
    M /= divM
    for ii in range(M.shape[1]-1):
        for jj in range(ii + 1, M.shape[1]):
            factM = np.dot(M[:, jj], M[:, ii])
            if np.any(factM):
                factM = np.sum(factM) / np.sum(M[:, ii] != 0)
            else:
                factM = 0
            M[:, jj] -= M[:, ii] * factM
            
    normM = np.linalg.norm(M, axis=0)
    M = M[:, normM >= tol]
    # M = M[:, normM >= 0.1]

    divM = np.array([M[:, idx][np.argmax(np.abs(M[:, idx]))] for idx in range(M.shape[1])])
    M /= divM

    max_abs_indices = np.argmax(np.abs(M), axis=0)
    divM = M[max_abs_indices, np.arange(M.shape[1])]

    M /= divM
    
    M = M[:, ::-1]
    for ii in range(M.shape[1]-1):
        for jj in range(ii + 1, M.shape[1]):
            factM = np.dot(M[:, jj], M[:, ii])
            if np.any(factM):
                factM = np.sum(factM) / np.sum(M[:, ii] != 0)
            else:
                factM = 0
            M[:, jj] -= M[:, ii] * factM

    divM = np.array([M[:, idx][np.argmax(np.abs(M[:, idx]))] for idx in range(M.shape[1])])
    
    M /= divM
    M  = M.reshape(3, 3, -1, order='F')

    asym = np.array([np.sum((M[:, :, idx] - M[:, :, idx].T)**2) > tol**2 * 9 for idx in range(M.shape[2])])
    aIdx = np.where(asym)[0]

    if asym.any():
        aSort = np.argsort(np.abs(M[1, 2, asym]) + np.abs(M[0, 2, asym])*10 + np.abs(M[0, 1, asym])*100)
        M[:, :, aIdx] = M[:, :, aIdx[aSort]]

    for idx in aIdx:
        signM = np.sign(M[1, 2, idx]*100 + M[2, 0, idx]*10 + M[0, 1, idx])
        M[:, :, idx] *= signM

    M = np.concatenate((M[:, :, ~asym], M[:, :, asym]), axis=2)
    M = np.round(M * 1e12) / 1e12
    # print(M)
    asym = np.concatenate([asym[~asym], asym[asym]])

    return M,asym

def point_Op(rotations,trans,r):
    is_point_op = np.full(len(rotations), False, dtype=bool)
    for i, rot in enumerate(rotations):
        new_r = np.dot(rot,r) % 1
        if np.allclose(new_r, r, atol=1e-5):
            is_point_op[i] = True
    pOp = np.zeros((3, 3,len(rotations)))
    for j in range(len(rotations)):
        pOp[:,:,j] = rotations[j]
    # return pOp
    return pOp[:, :, is_point_op]

def point_Op_in_xyz(lattice_matrix,point_rot):
    pOp_in_xyz = np.zeros((3,3,point_rot.shape[2]))
    for i in range(point_rot.shape[2]):
        # print("np.dot(point_rot[:,:,i] is \n", point_rot[:,:,i])
        # print("lattice_matrix",lattice_matrix)
        # print("dot result", np.dot(lattice_matrix.T,np.dot(point_rot[:,:,i],np.linalg.inv(lattice_matrix.T))))
        pOp_in_xyz[:,:,i] = np.dot(lattice_matrix,np.dot(point_rot[:,:,i],np.linalg.inv(lattice_matrix)))
    return pOp_in_xyz

def get_symm_anti_mat(aMat,aSym):
    bondIdx = True

    if bondIdx:
        aMatA = aMat[:, :, aSym]
    else:
        aMat = aMatS
        aMatA = np.zeros((3, 3, 0))
        aSym = np.zeros(9, dtype=bool)

    nSymMat = aMat.shape[2]
    dVect = np.transpose(np.array([aMatA[1, 2, :], aMatA[2, 0, :], aMatA[0, 1, :]])).T

    print(dVect)
   
    tol = 0.0001
    outprint = 0
    if outprint == 0:
        eStr = [["" for _ in range(3)] for _ in range(3)]
        first = np.ones((3, 3), dtype=bool)
        firstN = np.ones((3, 3), dtype=bool)
        nSymMat = aMatS.shape[2]

        for jj in range(3):
            for kk in range(3):
                for ii in np.where(np.abs(aMatS[jj, kk, :]) > tol)[0]:
                    prefix = ""
                    if firstN[jj, kk] and (aMatS[jj, kk, ii] > tol):
                        prefix = " "
                        firstN[jj, kk] = False
                    elif firstN[jj, kk]:
                        firstN[jj, kk] = False
                    
                    if (aMatS[jj, kk, ii] > tol) and not first[jj, kk]:
                        eStr[jj][kk] += "+"
                    elif first[jj, kk]:
                        first[jj, kk] = False

                    if np.abs(aMatS[jj, kk, ii] + 1) < tol:
                        eStr[jj][kk] += f"-{chr(65 + ii)}"
                    elif np.abs(aMatS[jj, kk, ii] - 1) < tol:
                        eStr[jj][kk] += f"{chr(65 + ii)}"
                    else:
                        eStr[jj][kk] += f"{aMatS[jj, kk, ii]:.3f}{chr(65 + ii)}"

        smatStr = '\n'.join(['|'.join(['|' + cell for cell in row]) + '|' for row in eStr])
        print("S = \n", smatStr)

    lStr = [len(e) for row in eStr for e in row]
    mStr = max(lStr)

    for jj in range(3):
        for kk in range(3):
            eStr[jj][kk] += ' ' * (mStr - len(eStr[jj][kk]))

    smatStr = ''
    for ii in range(3):
        if ii > 1:
            smatStr += '      '
        smatStr += '|' + '|'.join(eStr[ii]) + '|\n'

    aStr = ['' for _ in range(3)]
    first = [True] * 3
    firstN = [True] * 3

    for ii in range(3):
        for jj in np.where(np.abs(dVect[ii, :]) > tol)[0]:
            if firstN[jj] and (dVect[ii, jj] > tol):
                aStr[ii] += ' '
                firstN[jj] = False
            elif firstN[jj]:
                firstN[jj] = False

            if (dVect[ii, jj] > tol) and not first[ii]:
                aStr[ii] += '+'
            elif first[ii]:
                first[ii] = False

            if abs(dVect[ii, jj] + 1) < tol:
                aStr[ii] += '-D' + chr(48 + jj)
            elif abs(dVect[ii, jj] - 1) < tol:
                aStr[ii] += 'D' + chr(48 + jj)
            else:
                aStr[ii] += f"{dVect[ii, jj]:.2f}D{chr(48 + jj)}"

    lStr = [len(s) for s in aStr]
    mStr = max(lStr)

    for ii in range(3):
        aStr[ii] += ' ' * (mStr - len(aStr[ii]))

    amatStr = '[' + ','.join(aStr) + ']'
    print("Asymmetric Matrix Components D:\n", amatStr)
    
    return smatStr, amatStr

def map_atoms_to_supercell(unit_cell, supercell, index_displacements):
    mapped_indices = []
    for index, displacement in index_displacements:
        atom_coords = unit_cell[index].coords
        displacement_coords = np.dot(unit_cell.lattice.matrix.T,displacement)
        print(displacement_coords)
        new_coords = atom_coords + displacement_coords
        # print(new_coords)
        frac_in_supercell = np.dot(np.linalg.inv(supercell.lattice.matrix.T),new_coords)
        frac_in_supercell_new = np.mod(frac_in_supercell, 1.0)
        closest_index = None
        min_distance = float('inf')
        for i, atom in enumerate(supercell):
            distance = np.linalg.norm(frac_in_supercell_new - atom.frac_coords)
            # print(distance)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        mapped_indices.append((index, closest_index))
    return mapped_indices

def find_independent_combinations(aMat):
    n_slices = aMat.shape[2]
    indices = [(i, j) for i in range(3) for j in range(3)]

    # 遍历所有可能的组合，选择 n_slices 个不同的位置
    for combination in combinations(indices, n_slices):
        # 构建系数矩阵
        coeff_matrix = np.zeros((n_slices, n_slices))
        for row, (i, j) in enumerate(combination):
            coeff_matrix[row, :] = aMat[i, j, :]

        # 检查这个系数矩阵的秩是否等于 n_slices
        if np.linalg.matrix_rank(coeff_matrix) == n_slices:
            return combination, coeff_matrix

    return None, None  # 如果没有找到任何线性独立的组合

def get_all_combination(result_structure,symmetry_dataset,center_pos):
    all_combination = []
    for i in range(len(bonds_dict)):
        the_bond_index = i+1
        the_bond_subindex = 1
        dr = bonds_dict[the_bond_index][the_bond_subindex][2]
        lattice_matrix = result_structure.lattice.matrix.T
        product = lattice_matrix @ dr

        point_rot = point_Op(symmetry_dataset['rotations'],symmetry_dataset['translations'],center_pos[the_bond_index][the_bond_subindex])
        pOp = point_Op_in_xyz(lattice_matrix,point_rot)
        aMat, aSym = basic_sym_matrix(pOp, product, 1e-4)
        aMatS = aMat[:, :, ~aSym]
        combination, coeff_matrix = find_independent_combinations(aMatS)
        all_combination.append(combination)
    return all_combination

def flip_direction(direction):
    """
    Flip the sign of the first non-zero component in the direction.
    """
    # Split the direction string into components
    components = direction.split()
    # Flip the first non-zero component
    flipped = []
    flipped_sign = False
    for component in components:
        if not flipped_sign and int(component) != 0:
            flipped.append(str(-int(component)))
            flipped_sign = True
        else:
            flipped.append(component)
    return " ".join(flipped)

def generate_magmom_variants(start_direction, end_direction):
    variants = []
    # Original directions
    variants.append((start_direction, end_direction))
    # Flip end direction
    variants.append((start_direction, flip_direction(end_direction)))
    # Flip start direction
    variants.append((flip_direction(start_direction), end_direction))
    # Flip both directions
    variants.append((flip_direction(start_direction), flip_direction(end_direction)))
    return variants

def get_magmom_tags(start_atoms, end_atoms, all_combinations, total_atom_number, magnetic_atom_indices):
    magnetic_directions = {
        0: "2 0 0",
        1: "0 2 0",
        2: "0 0 2"
    }
    magmom_tags_list = []
    # Generate MAGMOM tags for each pair
    for i, (start_atom, end_atom) in enumerate(zip(start_atoms, end_atoms)):
        # Set default direction based on whether the atom is magnetic or not
        magmom_array = ["0 0 1" if idx in magnetic_atom_indices else "0 0 0" for idx in range(total_atom_number)]
        for combination in all_combinations[i]:
            start_direction = magnetic_directions[combination[0]]
            end_direction = magnetic_directions[combination[1]]
            # Generate variants
            variants = generate_magmom_variants(start_direction, end_direction) 
            for variant in variants:
                # Reset magmom_array to default before applying new variant
                magmom_array_temp = magmom_array.copy()
                magmom_array_temp[start_atom[1]] = variant[0]
                magmom_array_temp[end_atom[1]] = variant[1]
                magmom_tags_list.append(' '.join(magmom_array_temp))
    return magmom_tags_list

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_support_files(source_dir, target_dir, files):
    for file in files:
        src_file_path = os.path.join(source_dir, file)
        if os.path.exists(src_file_path):
            shutil.copy(src_file_path, target_dir)

def generate_magmom_files(base_dir, start_atoms, end_atoms, all_combinations, total_atom_number, magnetic_directions, template_path, source_dir, magnetic_atom_indices):
    with open(template_path, 'r') as file:
        template_content = file.readlines()
    support_files = ['KPOINTS', 'POTCAR', 'POSCAR', 'subjob']
    for i, (start_atom, end_atom) in enumerate(zip(start_atoms, end_atoms)):
        bond_dir = os.path.join(base_dir, f"bonds_with_diff_sym{i+1}")
        ensure_dir(bond_dir)
        for j, combination in enumerate(all_combinations[i]):
            combination_dir = os.path.join(bond_dir, f"diff_J{j+1}")
            ensure_dir(combination_dir)
            # Set default direction based on whether the atom is magnetic or not
            magmom_array = ["0 0 5" if idx in magnetic_atom_indices else "0 0 0" for idx in range(total_atom_number)]
            variants = generate_magmom_variants(magnetic_directions[combination[0]], magnetic_directions[combination[1]])
            for k, variant in enumerate(variants):
                variant_dir = os.path.join(combination_dir, str(k+1))
                ensure_dir(variant_dir)
                magmom_array[start_atom[1]] = variant[0]
                magmom_array[end_atom[1]] = variant[1]
                magmom_string = ' '.join(magmom_array)
                modified_content = [line if not line.strip().startswith('M_CONSTR') else f"M_CONSTR = {magmom_string}\n" for line in template_content]
                modified_content.append(f"\nMAGMOM = {magmom_string}\n")
                with open(os.path.join(variant_dir, "INCAR"), 'w') as f:
                    f.writelines(modified_content)
                copy_support_files(source_dir, variant_dir, support_files)

radius = 4
cif_file_path = r'/Users/haowang/Documents/Crystal structure/NiO.cif'

symmetry_dataset, result_structure = parse_and_symmetrize_structure(cif_file_path)
structure = deepcopy(result_structure)
bonds_list, total_neighbors,super_cell_scaling = find_and_store_bonds(structure, radius)
verification_result = verify_bonds_count(structure, bonds_list, total_neighbors)
sorted_bonds = assign_idx_and_sort_bonds(bonds_list)
sym_map_dict = atom_symmetry_mapping(symmetry_dataset['std_positions'],symmetry_dataset['rotations'],symmetry_dataset['translations'])
classified_groups = classify_bonds_by_symmetry(sorted_bonds, sym_map_dict,symmetry_dataset)
print_grouped_bonds(classified_groups)
# print(classified_groups)
bonds_dict = reset_idx_subidx_dict(classified_groups)
center_pos = center_of_bond(structure,bonds_dict)
# print(np.shape(center_pos[4][1]))
print(result_structure.lattice)
print(bonds_dict)

the_bond_index = 1
the_bond_subindex = 24
dr = bonds_dict[the_bond_index][the_bond_subindex][2]
lattice_matrix = result_structure.lattice.matrix.T
product = lattice_matrix @ dr

for the_bond_index,sub_bonds_dict in bonds_dict.items():
    for the_bond_subindex, bond_properties in sub_bonds_dict.items():
        
        print(f"键长第{the_bond_index}组 一共有个键")
        point_rot = point_Op(symmetry_dataset['rotations'],symmetry_dataset['translations'],center_pos[the_bond_index][the_bond_subindex])
        pOp = point_Op_in_xyz(lattice_matrix,point_rot)
        aMat, aSym = basic_sym_matrix(pOp, product, 1e-4)

        # print("aMat is", aMat)
        # print("aSym is", aSym)

        aMatS = aMat[:, :, ~aSym]

        get_symm_anti_mat(aMat,aSym)


point_rot = point_Op(symmetry_dataset['rotations'],symmetry_dataset['translations'],center_pos[the_bond_index][the_bond_subindex])
pOp = point_Op_in_xyz(lattice_matrix,point_rot)
aMat, aSym = basic_sym_matrix(pOp, product, 1e-4)

print("aMat is", aMat)
print("aSym is", aSym)

aMatS = aMat[:, :, ~aSym]

get_symm_anti_mat(aMat,aSym)

super_cell_scaling_list = list(super_cell_scaling)
super_cell_scaling_list = [x if x != 0 else 1 for x in super_cell_scaling_list]

scaling_matrix = np.array([[super_cell_scaling_list[0], 0, 0], [0, super_cell_scaling_list[1], 0], [ 0, 0,super_cell_scaling_list[2]]])

print(scaling_matrix)
supercell = result_structure.copy()
supercell.make_supercell(scaling_matrix)
# print(supercell)

supercell.to(filename=r"/Users/haowang/Documents/Codes/SymmMagHam/pyspinM/test6/POSCAR", fmt="poscar")

bond_end_index_in_supercell = []
for key, nested_dict in bonds_dict.items():
    for subkey, value in nested_dict.items():
        # if subkey == 1:
        if subkey == 1:
            bond_end_index_in_supercell.append((value[0],value[2]))  # 第四个元素的索引是3
bond_start_index_in_supercell = []
for key, nested_dict in bonds_dict.items():
    for subkey, value in nested_dict.items():
        # if subkey == 1:
        if subkey == 1:
            bond_start_index_in_supercell.append((value[0],np.array([0,0,0]))) 

print(bond_end_index_in_supercell)

bond_end_idx_in_sc    = map_atoms_to_supercell(result_structure,supercell,bond_end_index_in_supercell)
bond_start_idx_in_sc  = map_atoms_to_supercell(result_structure,supercell,bond_start_index_in_supercell)

cif_writer = CifWriter(supercell)
cif_writer.write_file('supercell.cif')

print(aMat.shape[2])

all_combination = get_all_combination(result_structure,symmetry_dataset,center_pos)
print(all_combination)
print(aMatS)
combination, coeff_matrix = find_independent_combinations(aMatS)

if combination:
    print("找到线性独立的元素组合位置:", combination)
    print("对应的系数矩阵:\n", coeff_matrix)
else:
    print("没有找到线性独立的元素组合")

print("bond_start_idx_in_sc",bond_start_idx_in_sc)
print("bond_end_idx_in_sc",bond_end_idx_in_sc)

magnetic_elements = {'Cu', 'Fe', 'Co', 'Cr', 'Mn', 'V', 'Ni'}
magnetic_atom_indices = [i for i, site in enumerate(supercell) if site.specie.symbol in magnetic_elements]

print("Magnetic atom indices:", magnetic_atom_indices)

total_atom_number = supercell.num_sites
magmom_tags = get_magmom_tags(bond_start_idx_in_sc, bond_end_idx_in_sc, all_combination, total_atom_number, magnetic_atom_indices)

base_dir = r"/Users/haowang/Documents/Codes/SymmMagHam/pyspinM/test6" 
template_path = r"/Users/haowang/Documents/Codes/SymmMagHam/pyspinM/test6/INCAR"  # 模板文件路径
source_dir = r"/Users/haowang/Documents/Codes/SymmMagHam/pyspinM/test6" 
magnetic_directions = {
    0: "5 0 0",
    1: "0 5 0",
    2: "0 0 5"
}
# Example usage
generate_magmom_files(base_dir, bond_start_idx_in_sc, bond_end_idx_in_sc, all_combination, total_atom_number, magnetic_directions, template_path, source_dir, magnetic_atom_indices)
# print(symmetry_dataset)