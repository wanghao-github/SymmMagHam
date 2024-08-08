### reading wyckoff label rewrite from Fortran 77 code
### author Hao Wang 2024/03/27 11:52-19:52  Darmstadt, Germany

import numpy as np
import re
from sympy import Matrix, symbols
from sympy import nsimplify


class SymmMagHamDict:
    uni_number = np.zeros((1651), dtype=int)
    nlabelparts_bns= np.zeros((1651,2), dtype=int)
    nlabelparts_og = np.zeros((1651,3), dtype=int)
    bnsog_point_op= np.zeros((1651,3,3), dtype=int)
    bnsog_origin= np.zeros((1651,3), dtype=int)
    bnsog_origin_denom= np.zeros((1651), dtype=int)
    ops_count= np.zeros((1651), dtype=int)
    wyckoff_site_count= np.zeros((1651), dtype=int)
    wyckoff_pos_count= np.zeros((1651,27), dtype=int)
    wyckoff_mult= np.zeros((1651,27), dtype=int)
    lattice_bns_vectors_count= np.zeros((1651), dtype=int)
    lattice_bns_vectors= np.zeros((1651,6,3), dtype=int)
    lattice_bns_vectors_denom= np.zeros((1651,6), dtype=int)
    ops_bns_point_op= np.zeros((1651,96), dtype=int)
    ops_bns_trans= np.zeros((1651,96,3), dtype=int)
    ops_bns_trans_denom= np.zeros((1651,96), dtype=int)
    ops_bns_timeinv= np.zeros((1651,96), dtype=int)
    wyckoff_bns_fract= np.zeros((1651,27,96,3), dtype=int)
    wyckoff_bns_fract_denom= np.zeros((1651,27,96), dtype=int)
    wyckoff_bns_xyz= np.zeros((1651,27,96,3,3), dtype=int)
    wyckoff_bns_mag= np.zeros((1651,27,96,3,3), dtype=int)
    lattice_og_vectors_count= np.zeros((1651), dtype=int)
    lattice_og_vectors= np.zeros((1651,6,3), dtype=int)
    lattice_og_vectors_denom= np.zeros((1651,6), dtype=int)
    ops_og_point_op= np.zeros((1651,96), dtype=int)
    ops_og_trans= np.zeros((1651,96,3), dtype=int)
    ops_og_trans_denom= np.zeros((1651,96), dtype=int)
    ops_og_timeinv= np.zeros((1651,96), dtype=int)
    wyckoff_og_fract = np.zeros((1651,27,96,3), dtype=int)
    wyckoff_og_fract_denom= np.zeros((1651,27,96), dtype=int)
    wyckoff_og_xyz= np.zeros((1651,27,96,3,3), dtype=int)
    wyckoff_og_mag= np.zeros((1651,27,96,3,3), dtype=int)

    point_op_label = []
    point_op_xyz = []
    point_op_matrix = np.zeros((48, 3, 3), dtype=int)
    point_op_hex_label = []
    point_op_hex_xyz = []
    point_op_hex_matrix = np.zeros((24,3, 3), dtype=int)

    nlabel_bns = np.empty(1651, dtype='U27')
    spacegroup_label_unified = np.empty(1651, dtype='U27')
    spacegroup_label_bns = np.empty(1651, dtype='U27')
    nlabel_og = np.empty(1651, dtype='U27')
    spacegroup_label_og= np.empty(1651, dtype='U27')
    wyckoff_label= np.empty((1651,27), dtype='U27')

    mag_symmety_data_bns = {}
    mag_symmety_data_og = {}

def read_mag_txt():
    # 打开数据文件
    with open(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\magnetic_data.txt', 'r') as file:
        # 读取非六角点操作符
        for i in range(48):
            n, label, xyz, *matrix_flat = file.readline().split()
            if int(n) != i + 1:
                raise Exception('error in numbering of nonhexagonal point operators')
            SymmMagHamDict.point_op_label.append(label)
            SymmMagHamDict.point_op_xyz.append(xyz)
            # print(matrix_flat)
            SymmMagHamDict.point_op_matrix[i] = np.array(matrix_flat, dtype=float).reshape(3, 3)
            # print(point_op_matrix[:,:,i])
        # 读取六角点操作符

        for i in range(24):
            n, label, xyz, *matrix_flat = file.readline().split()
            if int(n) != i + 1:
                raise Exception('error in numbering of hexagonal point operators')
            SymmMagHamDict.point_op_hex_label.append(label)
            SymmMagHamDict.point_op_hex_xyz.append(xyz)
            SymmMagHamDict.point_op_hex_matrix[i] = np.array(matrix_flat, dtype=float).reshape(3, 3)

        for i in range(1651):
            SymmMagHamDict.uni_number[i] = i+1
            line = file.readline().split()
            SymmMagHamDict.nlabelparts_bns[i,:] = line[0:2]  # 示例: 取前两个元素
            SymmMagHamDict.nlabel_bns[i] = line[2]
            SymmMagHamDict.spacegroup_label_unified[i] = line[3]
            SymmMagHamDict.spacegroup_label_bns[i] = line[4]
            SymmMagHamDict.nlabelparts_og[i,:] = line[5:8]
            SymmMagHamDict.nlabel_og[i] = line[8]
            SymmMagHamDict.spacegroup_label_og[i] = line[9]
            magtype = int(file.readline().strip())
            if magtype == 4:
                matrix_data = file.readline().split()
                SymmMagHamDict.bnsog_point_op[i,:,:] =  np.array(matrix_data[0:9], dtype=int).reshape(3, 3)
                SymmMagHamDict.bnsog_origin[i,:] = matrix_data[9:12]
                SymmMagHamDict.bnsog_origin_denom[i] = matrix_data[12]

            SymmMagHamDict.ops_count[i] = int(file.readline().strip())

            lines_data = []

            ops_lines = ((SymmMagHamDict.ops_count[i]-1)//4) + 1
            for num_line in range(ops_lines):
                line = file.readline().split()
                lines_data.append(line)

            for j in range(SymmMagHamDict.ops_count[i]): 
                SymmMagHamDict.ops_bns_point_op[i,j] = lines_data[j//4][0+6*(j%4)]
                SymmMagHamDict.ops_bns_trans[i,j,:] = lines_data[j//4][1+6*(j%4):4+6*(j%4)]
                SymmMagHamDict.ops_bns_trans_denom[i,j] = lines_data[j//4][4+6*(j%4)]
                SymmMagHamDict.ops_bns_timeinv[i,j] = lines_data[j//4][5+6*(j%4)]
            SymmMagHamDict.lattice_bns_vectors_count[i] = int(file.readline().strip())
            line = file.readline().split()

            for j in range(SymmMagHamDict.lattice_bns_vectors_count[i]):    
                SymmMagHamDict.lattice_bns_vectors[i,j,:] = line[0+4*j:3+4*j]
                SymmMagHamDict.lattice_bns_vectors_denom[i,j] = line[3+4*j]

            SymmMagHamDict.wyckoff_site_count[i] =  int(file.readline().strip())

            for j in range(SymmMagHamDict.wyckoff_site_count[i]):
                line = file.readline()
                matches = re.findall(r'[^"\s]\S*|"(?:\\.|[^"\\])*"', line)
                processed_matches = [match.strip('"') for match in matches]
                line = processed_matches
                SymmMagHamDict.wyckoff_pos_count[i,j],SymmMagHamDict.wyckoff_mult[i,j],SymmMagHamDict.wyckoff_label[i,j] = line
                for k in range(SymmMagHamDict.wyckoff_pos_count[i,j]):
                    line = file.readline().split()
                    SymmMagHamDict.wyckoff_bns_fract[i,j,k,:] = line[0:3]
                    SymmMagHamDict.wyckoff_bns_fract_denom[i,j,k] = line[3]
                    SymmMagHamDict.wyckoff_bns_xyz[i,j,k,:,:] = np.array(line[4:13], dtype=int).reshape(3, 3).transpose()
                    SymmMagHamDict.wyckoff_bns_mag[i,j,k,:,:] = np.array(line[13:22], dtype=int).reshape(3, 3).transpose()

            if magtype == 4:           
                SymmMagHamDict.ops_count[i] = int(file.readline().strip())
                lines_data = []
                ops_lines = (SymmMagHamDict.ops_count[i]-1)//4 + 1
                for num_line in range(ops_lines):
                    line = file.readline().split()
                    lines_data.append(line)

                for j in range(SymmMagHamDict.ops_count[i]): 
                    
                    SymmMagHamDict.ops_og_point_op[i,j] = lines_data[j//4][0+6*(j%4)]
                    SymmMagHamDict.ops_og_trans[i,j,:] = lines_data[j//4][1+6*(j%4):4+6*(j%4)]
                    SymmMagHamDict.ops_og_trans_denom[i,j] = lines_data[j//4][4+6*(j%4)]
                    SymmMagHamDict.ops_og_timeinv[i,j] = lines_data[j//4][5+6*(j%4)]

                SymmMagHamDict.lattice_og_vectors_count[i] = int(file.readline().strip())

                line = file.readline().split()
                for j in range(SymmMagHamDict.lattice_og_vectors_count[i]):    
                    SymmMagHamDict.lattice_og_vectors[i,j,:] = line[0+4*j:3+4*j]
                    SymmMagHamDict.lattice_og_vectors_denom[i,j] = line[3+4*j]

                SymmMagHamDict.wyckoff_site_count[i] =  int(file.readline().strip())

                for j in range(SymmMagHamDict.wyckoff_site_count[i]):

                    line = file.readline()
                    matches = re.findall(r'[^"\s]\S*|"(?:\\.|[^"\\])*"', line)
                    processed_matches = [match.strip('"') for match in matches]
                    line = processed_matches
                    SymmMagHamDict.wyckoff_pos_count[i,j],SymmMagHamDict.wyckoff_mult[i,j],SymmMagHamDict.wyckoff_label[i,j] = line

                    for k in range(SymmMagHamDict.wyckoff_pos_count[i,j]):
                        line = file.readline().split()
                        SymmMagHamDict.wyckoff_og_fract[i,j,k,:] = line[0:3]
                        SymmMagHamDict.wyckoff_og_fract_denom[i,j,k] = line[3]
                        SymmMagHamDict.wyckoff_og_xyz[i,j,k,:,:] =  np.array(line[4:13], dtype=int).reshape(3, 3).transpose()
                        SymmMagHamDict.wyckoff_og_mag[i,j,k,:,:] =  np.array(line[13:22], dtype=int).reshape(3, 3).transpose()
                        # print("wyckoff_og_xyz is",wyckoff_og_xyz[i,j,k,:,:])

def bns_symm_dictionary(uni_number=None):
    range_start = uni_number - 1  if uni_number is not None else 0
    range_end = uni_number if uni_number is not None else 1651

    
    for i in range(range_start, range_end):
        # key = SymmMagHamDict.nlabelparts_og[i, 2]
        key = SymmMagHamDict.uni_number[i]
        
        temp_bns_number =[]
        temp_operators_label = []
        temp_operators_matrix = []
        temp_lattice_bns_vectors_count = []
        temp_lattice_bns_vectors = []
        temp_lattice_bns_vectors_denom = []
        temp_lattice_bns_vectors_shift = []
        
        for j in range(SymmMagHamDict.ops_count[i]):
            if int(SymmMagHamDict.nlabelparts_bns[i, 0]) >= 168 and int(SymmMagHamDict.nlabelparts_bns[i, 0]) <= 194:
                op_label = SymmMagHamDict.point_op_hex_label[int(SymmMagHamDict.ops_bns_point_op[i, j]) - 1]
                op_matrix = SymmMagHamDict.point_op_hex_matrix[int(SymmMagHamDict.ops_bns_point_op[i, j]) - 1]
            else:
                op_label = SymmMagHamDict.point_op_label[int(SymmMagHamDict.ops_bns_point_op[i, j]) - 1]
                op_matrix = SymmMagHamDict.point_op_matrix[int(SymmMagHamDict.ops_bns_point_op[i, j]) - 1]
            temp_operators_label.append(op_label)
            temp_operators_matrix.append(op_matrix)
            
        temp_bns_number.append(SymmMagHamDict.nlabel_bns[i])
        temp_lattice_bns_vectors_count.append(SymmMagHamDict.lattice_bns_vectors_count[i])       
        temp_lattice_bns_vectors.append(SymmMagHamDict.lattice_bns_vectors[i,:])     
        temp_lattice_bns_vectors_denom.append(SymmMagHamDict.lattice_bns_vectors_denom[i])
        denom_reshaped = SymmMagHamDict.lattice_bns_vectors_denom[i].reshape(-1, 1)
        
        # temp_lattice_bns_vectors_shift.append(SymmMagHamDict.lattice_bns_vectors[i,:]/denom_reshaped)
        # print(denom_reshaped[2])
        # number_of_extra_bns_vector = SymmMagHamDict.lattice_bns_vectors_count[i]-3  #排除了前三个100 010 001的平移矢量 这个数最大是6实际上只有 最后的可能的三个数表示了平移
        # for j in range(number_of_extra_bns_vector):
        # the_j_extra_shift_vector = SymmMagHamDict.lattice_bns_vectors[i,3+j]
        safe_division_result = np.where(denom_reshaped == 0, 0, SymmMagHamDict.lattice_bns_vectors[i,:] / denom_reshaped)
        # 将处理后的结果添加到列表中
        temp_lattice_bns_vectors_shift.append(safe_division_result)
        # for j in range(6):
            # if denom_reshaped[j] != 0:
                # temp_lattice_bns_vectors_shift.append(SymmMagHamDict.lattice_bns_vectors[i,:]/denom_reshaped[j])
        
        if key not in SymmMagHamDict.mag_symmety_data_bns:
            SymmMagHamDict.mag_symmety_data_bns[key] = {
                'bns_number' : [],
                'operators_label': [], 
                'operators_matrix': [], 
                'operators_trans': [], 
                'ops_bns_timeinv': [],
                'lattice_bns_vectors_count':[],
                'lattice_bns_vectors': [],
                'lattice_bns_vectors_denom':[],
                'lattice_bns_vectors_shift':[],
                'wyckoffs': []
                }
        SymmMagHamDict.mag_symmety_data_bns[key]['bns_number'] = temp_bns_number
        SymmMagHamDict.mag_symmety_data_bns[key]['operators_label'] = temp_operators_label
        SymmMagHamDict.mag_symmety_data_bns[key]['operators_matrix'] = temp_operators_matrix
        SymmMagHamDict.mag_symmety_data_bns[key]['lattice_bns_vectors_count'] = temp_lattice_bns_vectors_count
        SymmMagHamDict.mag_symmety_data_bns[key]['lattice_bns_vectors'] = temp_lattice_bns_vectors
        SymmMagHamDict.mag_symmety_data_bns[key]['lattice_bns_vectors_denom'] = temp_lattice_bns_vectors_denom
        SymmMagHamDict.mag_symmety_data_bns[key]['lattice_bns_vectors_shift'] = temp_lattice_bns_vectors_shift
        
        temp_operators_trans = []
        temp_ops_bns_timeinv = []
        temp_wyckoff_bns_fract = []
        temp_wyckoff_bns_fract_denom = []
        for j in range(SymmMagHamDict.ops_count[i]):
            temp_operators_trans.append(SymmMagHamDict.ops_bns_trans[i, j, :] / int(SymmMagHamDict.ops_bns_trans_denom[i, j]))
            temp_ops_bns_timeinv.append(SymmMagHamDict.ops_bns_timeinv[i, j])
        
        SymmMagHamDict.mag_symmety_data_bns[key]['operators_trans'] = temp_operators_trans
        SymmMagHamDict.mag_symmety_data_bns[key]['ops_bns_timeinv'] = temp_ops_bns_timeinv

        SymmMagHamDict.mag_symmety_data_bns[key]['wyckoff_bns_fract'] = temp_wyckoff_bns_fract
        SymmMagHamDict.mag_symmety_data_bns[key]['wyckoff_bns_fract_denom'] = temp_wyckoff_bns_fract_denom
        wyckoffs_list = []
        
        for j in range(SymmMagHamDict.wyckoff_site_count[i]):

            wyckoff_info = {
                'wyckoff_mult': [],
                'wyckoff_label': [],
                'wyckoff_bns_fract' : [],
                'wyckoff_bns_fract_denom' : [],
                'wyckoff_bns_xyz' : [],
                'wyckoff_bns_mag' : [],
                'fraction_xyz_shift': []
            }
            mult = SymmMagHamDict.wyckoff_mult[i,j]
            label = SymmMagHamDict.wyckoff_label[i,j]
            wyckoff_info['wyckoff_mult'].append(mult)
            wyckoff_info['wyckoff_label'].append(label)
            
            for k in range(SymmMagHamDict.wyckoff_pos_count[i, j]):  # 假设coordinate_count[i, j]是每个位点坐标的数量
                
                coordinate_xyz = SymmMagHamDict.wyckoff_bns_xyz[i, j, k]  # 获取坐标的示例方法
                coordinate_mag = SymmMagHamDict.wyckoff_bns_mag[i, j, k]
                bns_xyz_shift =  SymmMagHamDict.wyckoff_bns_fract[i,j,k] / SymmMagHamDict.wyckoff_bns_fract_denom[i,j,k]
                
                # fraction_values = str([nsimplify(value, tolerance=0.01) for value in bns_xyz_shift])
                fraction_values = ', '.join(str(nsimplify(value, tolerance=0.01)) for value in bns_xyz_shift)
                # print(fraction_values)
                
                x, y, z = symbols('x y z')
                sympy_matrix = Matrix(coordinate_xyz)
                vector = Matrix([x, y, z])
                result = sympy_matrix * vector
                result_str = ",".join([str(elem) for elem in result])

                mx, my, mz = symbols('mx my mz')
                sympy_matrix = Matrix(coordinate_mag)
                vector = Matrix([mx, my, mz])
                result_mag = sympy_matrix * vector
                result_mag_str = ",".join([str(elem) for elem in result_mag])
                
                # print(result_str)
                variables = result_str.split(",")
                fractions = fraction_values.split(",")
                # print(fractions)
                
                
                result_shift_str = ','.join([f"{var}+{frac}" if frac != '0' else var for var, frac in zip(variables, fractions)])
                result_shift_str = result_shift_str.replace('+-', '-')

                wyckoff_info['wyckoff_bns_xyz'].append(result_str)
                wyckoff_info['wyckoff_bns_mag'].append(result_mag_str)
                wyckoff_info['fraction_xyz_shift'].append(result_shift_str)


                
            wyckoffs_list.append(wyckoff_info)

        SymmMagHamDict.mag_symmety_data_bns[key]['wyckoffs']= wyckoffs_list
        
    return SymmMagHamDict.mag_symmety_data_bns
        # print(mag_symmety_data_bns[key])

def gen_mag_table():
    with open(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\gen_og_table.txt', 'w') as file:
        for i in range(1651):        
            label_bns_str = ' '.join(map(str, SymmMagHamDict.nlabelparts_bns[i]))
            label_og_str = ' '.join(map(str, SymmMagHamDict.nlabelparts_og[i]))
            spacegroup_label_unified_str = ' '.join(map(str,SymmMagHamDict.spacegroup_label_unified[i]))
            # 组合最终的字符串
            line = f"BNS {label_bns_str} OG {label_og_str} UNI_label {spacegroup_label_unified_str}\n"
            # 写入文件
            file.write(line)
            for j in range(SymmMagHamDict.ops_count[i]):
                
                op_label = str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['operators_label'][j])
                op_trans = str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['operators_trans'][j])
                op_timeinv = str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['ops_bns_timeinv'][j])
                line = f"'operators_label':{op_label},   op_trans:{op_trans} time_inv: {op_timeinv}\n"

            # 写入文件
                file.write(line)
            # print(str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs']))
            line = str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['lattice_bns_vectors_shift'][0][3:6,:]) + '\n'
            file.write(line)
            
            for j in range(SymmMagHamDict.wyckoff_site_count[i]):
                for k in range(SymmMagHamDict.wyckoff_pos_count[i,j]):
                #     print(k)
                #     # print(str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs']['wyckoff_bns_xyz'][k]))
                    line = str(SymmMagHamDict.wyckoff_pos_count[i,j]) + ' ' \
                            + str(SymmMagHamDict.wyckoff_mult[i,j]) + ' ' \
                            + str(SymmMagHamDict.wyckoff_label[i,j]) + ' ' \
                            + str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['wyckoffs'][j]['fraction_xyz_shift'][k])+ ' '  \
                            + str(SymmMagHamDict.mag_symmety_data_bns[SymmMagHamDict.nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_bns_mag'][k])+   '\n'
                            # + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_mult']) + ' ' \
                            # + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_label'])+ ' ' \               

                    file.write(line)
                      
read_mag_txt()
# aa = SymmMagHamDict.wyckoff_mult.max()
with open(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\oglabel.txt', 'w') as file:
    for j in range(1651):
        file.write(str(SymmMagHamDict.uni_number[j]))
        file.write('\n')
# print(SymmMagHamDict.nlabelparts_og)
a = bns_symm_dictionary(1620)
print(SymmMagHamDict.mag_symmety_data_bns)                  
# if __name__ == "__main__":
#     # 仅当该脚本被直接运行时执行的代码
#     # a = SymmMagHamDict()
#     read_mag_txt()
    
    # 调用方法1  类的属性
    # bns_symm_dictionary(1)  # 例子
    # a = SymmMagHamDict.mag_symmety_data_bns
    # print(a)
    
    # 调用方法2  函数返回值
    # AA = bns_symm_dictionary(1)
    # print(AA)
    
    
# read_mag_txt()
# bns_symm_dictionary()     
# print(mag_symmety_data_bns)
# gen_mag_table()