import numpy as np
import ase 
import spglib as spg
import pymatgen
from math import *
from pymatgen.io.cif import CifParser
from spglib import get_spacegroup
import re
from read_mag_data import bns_symm_dictionary
from read_mag_data import read_mag_txt
from read_mag_data import SymmMagHamDict
import read_cif
from pymatgen.core import Structure, Lattice
from copy import deepcopy
from bonds import Bond
import sympy as sp


def mod_one(arr):
    return np.mod(arr, 1)

mag_uni_number = int(read_cif.mag_dataset['uni_number'])

print(mag_uni_number)

# mag_uni_number = 1620
#read_cif.mag_dataset['uni_number']

# number_of_equal_site =  read_cif.mag_dataset['std_types']

equal_site_start = 0
number_of_equal_site = 2
print(number_of_equal_site)

std_positions =read_cif.mag_dataset['std_positions']
std_lattice =read_cif.mag_dataset['std_lattice']
print(std_positions)
# read_cif.mag_dataset['uni_number']
# print(read_cif.dataset)
# print(uni)
# read_mag_txt()

# og_key = int(read_cif.get_msg_numbers()[mag_uni_number-1][2].strip().split('.')[-1])
# print(og_key)
bns_op_dict = bns_symm_dictionary(mag_uni_number)
print(bns_op_dict[mag_uni_number]['wyckoffs'][0]['fraction_xyz_shift'])
space_group_numb = int(read_cif.get_msg_numbers()[mag_uni_number-1][1].strip().split('.')[0])
# space_group_numb = int(bns_op_dict[mag_uni_number]['bns_number'][0].strip('"').split(".")[0])
print(space_group_numb)
# lattice = read_cif.build_prototype(space_group_numb)

# wyckoff_mult =  SymmMagHamDict.wyckoff_mult[399]
# wyckoff_xyz_op_matrix=SymmMagHamDict.wyckoff_bns_xyz[399,0,1]

# print(wyckoff_mult)
# # print(lattice)
# indices_of_fours = np.where(wyckoff_mult == 4)

# print("值为4的元素的索引", indices_of_fours[0])
#这个[0]指的是元组的第一个元素 因为返回一个一维数组的元组

# for i in indices_of_fours[0]:
#     print(i)
    
# mag_uni_number = 1620
# count_fours = bbb.count(4)
# found_match = False
for j in range(SymmMagHamDict.wyckoff_site_count[mag_uni_number-1]):
    # 对符合条件的操作进行处理
    if SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j] == number_of_equal_site:
        print(SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j])
        print(SymmMagHamDict.wyckoff_label[mag_uni_number-1,j])
        
        # 检查简并度
        need_verified_mult = SymmMagHamDict.wyckoff_pos_count[mag_uni_number-1,j] * (SymmMagHamDict.lattice_bns_vectors_count[mag_uni_number-1] - 3 + 1)
        diff = need_verified_mult - SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j]
        if diff == 0:   
            print("简并度没问题")
        else:
            print("简并度有问题")
            
        operations_success = [False] * SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j]
        # 对每个操作进行处理
        num_lattice_trans = SymmMagHamDict.lattice_bns_vectors_count[mag_uni_number-1] - 3 +1
        
        for m in range(num_lattice_trans):
            for k in range(SymmMagHamDict.wyckoff_pos_count[mag_uni_number-1,j]):
                rot_op = SymmMagHamDict.wyckoff_bns_xyz[mag_uni_number-1,j,k]
                tran_op = SymmMagHamDict.wyckoff_bns_fract[mag_uni_number-1,j,k] / SymmMagHamDict.wyckoff_bns_fract_denom[mag_uni_number-1,j,k]

                print(f"rot_op 是 {rot_op}")
                print(f"tran_op 是 {tran_op}")
                # print(SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['lattice_bns_vectors_shift']) ###键名不用减1
                if m == 0:
                    lattice_shift = np.array([0, 0, 0])
                else:
                    lattice_shift = SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['lattice_bns_vectors_shift'][0][m+2]
                # print(f"当前的lattice_bns_vectors_shift 是 {SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['lattice_bns_vectors_shift'][0][m+2]}")  
                new_pos = np.zeros((number_of_equal_site,3), dtype=float)
                # print(number_of_equal_site)
                print(f"当前的lattice_bns_vectors_shift 是{lattice_shift}")
                
                for n in range(number_of_equal_site):
                    new_pos[ k + m * SymmMagHamDict.wyckoff_pos_count[mag_uni_number-1,j],:] = \
                    np.dot(rot_op, std_positions[equal_site_start + n,:]) + tran_op + lattice_shift
                
                # lattice_trans = SymmMagHamDict.
                # print(new_pos)
                print(f"当前的equal_site_start 是{equal_site_start}")
                print(f"number_of_equal_site is {number_of_equal_site}")
                adjusted_std_positions = mod_one(std_positions[equal_site_start:(equal_site_start + number_of_equal_site),:])
                
                print(adjusted_std_positions)
                adjusted_new_pos = mod_one(new_pos)
                print(adjusted_new_pos)
                tolerance = 1e-4
                # matches = [False] * 4  # 初始化匹配结果
                operation_success = False
                # 进行匹配检查
                for i, new_point in enumerate(adjusted_new_pos):
                    for orig_point in adjusted_std_positions:
                        if np.all(np.abs(new_point - orig_point) < tolerance):
                            # matches[i] = True
                            # found_match = True
                            operation_success = True
                            break  # 找到匹配后不再继续搜索
                    if operation_success:
                        break  # 如果当前操作成功，停止检查当前操作的其他点         
                operations_success[k + m * SymmMagHamDict.wyckoff_pos_count[mag_uni_number-1,j]] = operation_success

            print(operations_success)

            if all(operations_success):
                matched_wyckoff_mult = SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j]
                matched_wyckoff_label = SymmMagHamDict.wyckoff_label[mag_uni_number-1,j]
                # matched_position = 
                print(f"对于j={j}, 所有操作都至少有一个点成功匹配。")        
                print(f"这个体系的wyckoff_label是{SymmMagHamDict.wyckoff_label[mag_uni_number-1,j]}")
            else:
                print(f"没找到对称性操作") 



             
og_number  = read_cif.get_msg_numbers()[mag_uni_number-1][2].strip().split('.')[-1]
print(f"og_number is {og_number}")

print(f"匹配的wyckoff位置是{matched_wyckoff_mult}{matched_wyckoff_label}")

spg_number = int(read_cif.get_msg_numbers()[mag_uni_number-1][1].strip().split('.')[0])
print(f"spg_number is {spg_number}")
print(f"uni_number is {mag_uni_number}")





# lattice = read_cif.build_prototype(spg_number)
lattice =std_lattice
print(lattice)
position = std_positions[equal_site_start:(equal_site_start + number_of_equal_site),:]

# species = ["Mn","Mn","Mn","Mn"]

species = ["Cr","Cr"]
primitive_cell = Structure(lattice, species,position)


# atom_numbers= [24,24,24,24]
# primitive_cell_for_spg = (lattice, position,atom_numbers)
# print(primitive_cell_for_spg)


# print(SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'][1]['wyckoff_mult'])

# print(SymmMagHamDict.wyckoff_label[mag_uni_number])
lattice = primitive_cell.lattice
a, b, c = lattice.a, lattice.b, lattice.c

radius = 3.3
scaling_factors = [
    max(2, int(2 * radius / a)),
    max(2, int(2 * radius / b)),
    max(2, int(2 * radius / c))
]

structure = deepcopy(primitive_cell)
scaling_matrix = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
supercell = structure.make_supercell(np.array(scaling_matrix).diagonal())

# print(primitive_cell)



# 构建超胞
# supercell = primitive_cell.make_supercell(scaling_factors)

# 设定搜索半径（根据你的系统调整，一般为最大可能键长）
# radius = 10.0  # 单位为Å（埃）

# 获取所有邻居
each_atom_neighbors = supercell.get_all_neighbors(radius, include_index=True)

radius = 6
# 创建一个映射来记录每个超胞原子对应的原胞原子编号
print(supercell)
print(primitive_cell)
# 遍历超胞中的每个原子
def super_to_origin_pos(supercell):
    mapping = {}
    for idx, atom in enumerate(supercell):
        original_pos = (atom.frac_coords * np.array(scaling_matrix).diagonal()) % 1
        original_idx = min(range(len(primitive_cell)), key=lambda i: np.sum((primitive_cell[i].frac_coords - original_pos)**2))
        # original_idx = min(range(len(primitive_cell)), key=lambda i: np.sum((np.minimum(np.abs(primitive_cell[i].frac_coords - original_pos), 1 - np.abs(primitive_cell[i].frac_coords - original_pos)))**2))
        mapping[idx] = original_idx
    # for k, v in mapping.items():
        # print(f"Supercell atom {k} corresponds to primitive cell atom {v}")
    return mapping


# bond_info = Bond(supercell)
# print(bond_info.distance_matrix[1,1])
# super_to_origin_dict = super_to_origin_pos(supercell)
# print(super_to_origin_dict[13])


mapping = super_to_origin_pos(supercell)
distance_matrix = supercell.distance_matrix

# 打印所有键的信息
# print("Supercell Bonds:")
# for i in range(len(distance_matrix)):
#     for j in range(i + 1, len(distance_matrix)):  # 避免重复和自身比较
#         if distance_matrix[i, j] > 0:  # 选取非零距离的原子对
#             print(f"Supercell atom pair ({i}, {j}) - distance: {distance_matrix[i, j]:.3f} Å")
#             print(f"    Corresponds to primitive cell atom pair ({mapping[i]}, {mapping[j]})")

def preprocess_expression(expr):
    """预处理表达式，确保负变量被正确处理为乘以 -1 的形式。"""
    # 替换 -mx 和 -mz 为 -1*mx 和 -1*mz
    expr = re.sub(r'-(mx)', r'-1*\1', expr)
    expr = re.sub(r'-(my)', r'-1*\1', expr)
    expr = re.sub(r'-(mz)', r'-1*\1', expr)
    return expr

# def preprocess_expression(expr):
#     """预处理表达式，确保负变量被正确处理为乘以 -1 的形式，并用括号包围。"""
#     # 替换 -mx, -my, -mz 为 (-1)*mx, (-1)*my, (-1)*mz
#     patterns = ['mx', 'my', 'mz']
#     for pattern in patterns:
#         expr = re.sub(rf'-(?={pattern})', r'(-1)*', expr)
#     return sp.sympify(expr)


def get_direction_vector(site1, site2, lattice):
    """
    计算从site1到site2的方向矢量,考虑周期性边界。
    """
    vec = site2.frac_coords - site1.frac_coords
    vec = vec - np.round(vec)  # 考虑周期性边界
    return lattice.get_cartesian_coords(vec)

# for i in range(len(distance_matrix)):
#     for j in range(i + 1, len(distance_matrix)):  # 避免重复和自身比较
#         if radius > distance_matrix[i, j] > 0:  # 可能需要设置一个更实际的阈值，比如键长上限
#             bonds.append((distance_matrix[i, j], i, j, mapping[i], mapping[j]))

target_wyckoff_label = str(matched_wyckoff_mult)+str(matched_wyckoff_label)
target_wyckoff_label=target_wyckoff_label.strip()
for j in range(len(SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'])):
    mult = str(SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'][j]['wyckoff_mult'][0])
    label = str(SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'][j]['wyckoff_label'][0])
    # print(mult,label)
    wyckoff_result = str(mult) + label.strip()
    wyckoff_result=wyckoff_result.strip()
    if str(target_wyckoff_label) == wyckoff_result:
        wyckoff_mag= SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'][j]['wyckoff_bns_mag']
        wyckoff_shift_xyz = SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['wyckoffs'][j]['fraction_xyz_shift']
        print(position)

# order = [0, 2, 1, 3]

order = [0, 1]
reordered_expressions = [wyckoff_shift_xyz[i] for i in order]
reordered_mag = [wyckoff_mag[i] for i in order]

print(reordered_mag)

def get_wyckoff_mag_from_ori_pos(ori_pos,reordered_mag):
    return reordered_mag[ori_pos]


bonds = []
directions = []
#这个地方的i是超胞中每个原子的编号
for i, site in enumerate(supercell):
    
    # ori_pos=super_to_origin_pos(supercell)[i]
    
    # print("超胞中编号",i,"的原子 在单胞中编号为",ori_pos)
    neighbors = supercell.get_neighbors(site, r=radius)
    # print(neighbors)
    # frac_coords_list = [neighbor.frac_coords for neighbor in neighbors]
    # print(frac_coords_list)
    print(neighbors)
    print("现在是第",i,"个原子,坐标是",site.frac_coords)
    for neighbor in neighbors:
        print(neighbor.index)
        # print(neighbor)
        # if neighbor.index > i:  # 避免重复 这个index指的是与当前原子近邻的几个原子的编号 可以认为是末端原子编号 而i是起点原子编号 这样只记录了从小到大的键
            # print("现在是第",neighbor.index,"个原子","的第",i,"个键")
            # direction_vector = get_direction_vector(site, neighbor, supercell.lattice)
            # bonds.append((neighbor.nn_distance, i, neighbor.index,mapping[i],mapping[neighbor.index],direction_vector))
        
        direction_vector = get_direction_vector(site, neighbor, supercell.lattice)
        norm_direction = np.linalg.norm(direction_vector)
        if norm_direction > 0:  # 确保不是零向量
            direction_vector /= norm_direction  # 标准化方向向量
            
        if neighbor.index > i:
            directions.append(direction_vector)
            start_point_mag = get_wyckoff_mag_from_ori_pos(mapping[i],reordered_mag)
            end_point_mag = get_wyckoff_mag_from_ori_pos(mapping[neighbor.index],reordered_mag)
            
            start_point_mag_modified = preprocess_expression(start_point_mag)
            end_point_mag_modified = preprocess_expression(end_point_mag)
            
            print("start_point_mag is",start_point_mag)
            print("end_point_mag is",end_point_mag)
            bonds.append((neighbor.nn_distance, i, neighbor.index,mapping[i],mapping[neighbor.index],\
                direction_vector,start_point_mag,end_point_mag))
        elif neighbor.index == i:  # 自环，检查方向
            # 检查是否有相反的方向已经存在
            add_bond = True
            
            for dir_vec in directions:
                if np.dot(direction_vector, dir_vec) < -0.999:  # 方向近似相反
                    add_bond = True
                    break
            print(add_bond)        
            if add_bond:
                bonds.append((neighbor.nn_distance, i, neighbor.index, mapping[i], mapping[neighbor.index], \
                    direction_vector,start_point_mag,end_point_mag))
                directions.append(direction_vector)    
            # 检查是否已存在相反的方向          
        # elif neighbor.index == i
# 上面这种方法没有考虑到0-0 或者1-1这种情况


# all_neighbors_withsame = []  # 用于存储所有邻居信息，包括重复的

# for i, site in enumerate(supercell):
#     neighbors = supercell.get_neighbors(site, r=radius)
#     print("现在是第", i, "个原子, 坐标是", site.frac_coords)
#     for neighbor in neighbors:
#         print(neighbor.index)
#         all_neighbors_withsame.append((neighbor.nn_distance, i, neighbor.index, mapping[i], mapping[neighbor.index]))

# # 去重处理
# seen = set()
# for distance, start, end, orig_start, orig_end in all_neighbors_withsame:
#     if start > end:  # 确保小索引始终在前，大索引始终在后，这样可以一致化比较
#         start, end = end, start
#         orig_start, orig_end = orig_end, orig_start
#     if (start, end) not in seen:
#         seen.add((start, end))
#         bonds.append((distance, start, end, orig_start, orig_end))
        



# 按键长排序
# bonds.sort()
bonds.sort(key=lambda x: x[0])
tolerance = 0.001

all_neighbors = []
current_neighbor = []

# 初始化第一个分组
if bonds:
    current_neighbor.append(bonds[0])

for bond in bonds[1:]:
    if abs(bond[0] - current_neighbor[-1][0]) <= tolerance:
        current_neighbor.append(bond)
    else:
        all_neighbors.append(current_neighbor)
        current_neighbor = [bond]

# 添加最后一组
if current_neighbor:
    all_neighbors.append(current_neighbor)

# 分配紧邻级别并打印
for idx, group in enumerate(all_neighbors, start=1):
    print(f"Neighbor Level {idx}:")
    for bond in group:
        print(f"  Length: {bond[0]:.3f} Å, Supercell: {bond[1]} to {bond[2]}, Primitive: {bond[3]} to {bond[4]}")

# print(all_neighbors[0])

max_distance = np.max(primitive_cell.distance_matrix)
print("primitive_cell最大键长为:", max_distance, "Å")



for idx, group in enumerate(all_neighbors, start=1):
    average_length = np.mean([bond[0] for bond in group])  # 计算当前组所有键长的平均值
    print(f"Neighbor Level {idx}: Average Bond Length = {average_length:.3f} Å")
    if average_length <= max_distance+0.001:    
        print(group)


operators = {}

operators['rot'] = SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['operators_matrix']
operators['trans'] = SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['operators_trans']
operators['timeinv'] = SymmMagHamDict.mag_symmety_data_bns[mag_uni_number]['ops_bns_timeinv']
    
print(operators)

op_mapping_table = []

# 应用每个对称操作
for idx, rot in enumerate(operators['rot']):
    trans = operators['trans'][idx]
    for index, atom in enumerate(primitive_cell):
        original_pos = atom.frac_coords
        # print(original_pos)
        new_pos = np.dot(rot, original_pos) + trans  # 应用旋转和平移
        new_pos = new_pos % 1  # 考虑周期性边界条件
        # print(original_pos-new_pos)
        # print(atom.frac_coords - np.array(new_pos))
        # 查找最接近的原始原子
        distances = np.linalg.norm(primitive_cell.frac_coords - new_pos, axis=1)
        print(distances)
        closest_atom_index = np.argmin(distances)
        print(closest_atom_index)
        
        #这个地方的idx是对称性操作元素的指标 index是变换前的指标 closest_atom_index是变换后的指标
        op_mapping_table.append((idx, index, closest_atom_index))

# 打印映射表
for entry in op_mapping_table:
    if entry[1] == 0:
        print(f"under op No. {entry[0]} Original atom at index {entry[1]} maps to new position {entry[2]}")

def get_op_idx(in_pos,out_pos,op_mapping_table):
    group_op_list = []
    for i, map in enumerate(op_mapping_table):
        op_number,init_pos,mapped_pos = map
        if in_pos==init_pos and mapped_pos == out_pos:            
            group_op_list.append(op_number)
    return group_op_list



# print(all_neighbors[1])


# print(all_neighbors[1])
print(bonds)

all_op_number_list = []
for idx, bond in enumerate(all_neighbors[0]):
    bond_index1 = bond[3]
    bond_index2 = bond[4]
    operation_list = get_op_idx(bond_index1,bond_index2,op_mapping_table)
    all_op_number_list.append((bond, set(operation_list)))
 
from collections import defaultdict
operation_groups = defaultdict(list)

for bond, operations in all_op_number_list:
    operation_groups[frozenset(operations)].append(bond)

# 输出每组操作的键
for ops, bonds in operation_groups.items():
    print(f"Operations: {list(ops)}")
    for bond in bonds:
        print(f" Bond from {bond[1]} to {bond[2]} with length {bond[0]:.3f} Å")    
        
print(all_op_number_list)

Jxx =0
Jyy = 0
Jzz = 0
bond_parameters = []

my = sp.symbols('my')
nmy = sp.symbols('nmy')  # nmy 代表 -my

# 映射旧的符号表达方式到新的符号
symbol_map = {
    'my': my,
    'nmy': -my
}

for idx, bond in enumerate(all_neighbors[0]):
    bond_length = bond[0]
    start_i_supercell = bond[1]
    end_i_supercell = bond[2]
    start_i_primcell = bond[3]
    end_i_primcell = bond[4]
    direction_vector = bond[5]
    start_point_mag = str(bond[6])
    end_point_mag= str(bond[7])
    Jxx = sp.symbols(start_point_mag.split(',')[0]) * sp.symbols(end_point_mag.split(',')[0])
    Jyy = sp.symbols(start_point_mag.split(',')[1]) * sp.symbols(end_point_mag.split(',')[1])
    Jzz = sp.symbols(start_point_mag.split(',')[2]) * sp.symbols(end_point_mag.split(',')[2])
    bond_parameters.append((idx, Jxx, Jyy, Jzz))

# my = sp.symbols('my')
# expr1 = -my**2
# expr2 = -my*my

# if sp.simplify(expr1 - expr2) == 0:
#     print("表达式是等价的")
# else:
#     print("表达式不等价")
groups = {}

# 对每个bond参数进行比较，找出相等的组



for idx, Jxx, Jyy, Jzz in bond_parameters:
    found_group = False
    # 遍历现有的所有组来寻找匹配
    for group_idx, group_members in groups.items():
        # print("group_member is", group_members)
        # 检查当前bond是否与已有分组等价
        group_sample = group_members[0]
        # print("group_sample is", group_members[0])
        # 初始化匹配标志
        is_match = True
        # 逐一比较每个参数
        for current, sample in zip((Jxx, Jyy, Jzz), group_sample[1:]):
            print("current", current)
            print("sample", sample)
            # 使用equals方法来检查是否相等
            if not current.equals(sample):
                print("not Match")
                print("Difference:", current - sample)
                is_match = False
                break
        # 如果找到匹配的组
        if is_match:
            groups[group_idx].append((idx, Jxx, Jyy, Jzz))
            found_group = True
            break  # 这里的 break 只是退出当前对比，继续尝试下一个组

    # 如果遍历完所有组都没有找到匹配的组，则创建一个新的组
    if not found_group:
        groups[idx] = [(idx, Jxx, Jyy, Jzz)]

# 打印分组结果
for group_idx, group_members in groups.items():
    print(f"Group {group_idx}: Bonds {', '.join(str(b[0]) for b in group_members)}")

# 'wyckoff_bns_mag'
# 'fraction_xyz_shift'
# SymmMagHamDict.wyckoff_mult[mag_uni_number-1,j]
# print(a)
# def wyckoff_solver(wyckoff_pos,frac_pos):

# # 假设这些符号已经定义
# mx, my, mz = sp.symbols('mx my mz')

# # 定义测试值
# test_values = [
#     {mx: 1, my: 2, mz: 3},
#     {mx: -1, my: -2, mz: -3},
#     {mx: 0, my: 0.5, mz: 1.5}
# ]

# def are_expressions_equivalent(expr1, expr2, test_vals):
#     """使用数值替换测试表达式是否在多个点上等价。"""
#     for vals in test_vals:
#         if expr1.subs(vals).evalf() != expr2.subs(vals).evalf():
#             print(expr1.subs(vals).evalf())
#             print(expr2.subs(vals).evalf())
#             return False
#     return True

# # 循环逻辑，用于处理每个bond和分组
# for idx, Jxx, Jyy, Jzz in bond_parameters:
#     found_group = False
#     # 遍历现有的所有组来寻找匹配
#     for group_idx, group_members in groups.items():
#         group_sample = group_members[0]

#         # 初始化匹配标志
#         is_match = True

#         # 逐一比较每个参数
#         for current, sample in zip((Jxx, Jyy, Jzz), group_sample[1:]):
#             print("current", current)
#             print("sample", sample)
#             # 使用数值测试来检查是否相等
#             if not are_expressions_equivalent(current, sample, test_values):
#                 print("not Match")
#                 print("Difference at points:", current - sample)
#                 is_match = False
#                 break

#         # 如果找到匹配的组
#         if is_match:
#             groups[group_idx].append((idx, Jxx, Jyy, Jzz))
#             found_group = True
#             break  # 这里的 break 只是退出当前对比，继续尝试下一个组

#     # 如果遍历完所有组都没有找到匹配的组，则创建一个新的组
#     if not found_group:
#         groups[idx] = [(idx, Jxx, Jyy, Jzz)]

# # 打印分组结果
# for group_idx, group_members in groups.items():
#     print(f"Group {group_idx}: Bonds {', '.join(str(b[0]) for b in group_members)}")