### reading wyckoff label rewrite from Fortran 77 code
### author Hao Wang 2024/03/27 11:52-19:52  Darmstadt, Germany

import numpy as np
import re
from sympy import Matrix, symbols
from sympy import nsimplify


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
            point_op_label.append(label)
            point_op_xyz.append(xyz)
            # print(matrix_flat)
            point_op_matrix[i] = np.array(matrix_flat, dtype=float).reshape(3, 3)
            # print(point_op_matrix[:,:,i])
        # 读取六角点操作符

        for i in range(24):
            n, label, xyz, *matrix_flat = file.readline().split()
            if int(n) != i + 1:
                raise Exception('error in numbering of hexagonal point operators')
            point_op_hex_label.append(label)
            point_op_hex_xyz.append(xyz)
            point_op_hex_matrix[i] = np.array(matrix_flat, dtype=float).reshape(3, 3)

        # print(point_op_hex_label)
        # print(point_op_hex_xyz)
        # print(point_op_hex_matrix)
        # 读取每个磁性空间群的数据

        for i in range(1651):
            line = file.readline().split()
            # print(line)   
            # line_part = re.findall(r'".+?"|\S+', line)    
            # print('the first line here',line)
            nlabelparts_bns[i,:] = line[0:2]  # 示例: 取前两个元素
            # print('nlabelparts_bns[i,:] is ', line_part[0:2])
            nlabel_bns[i] = line[2]
            # print(nlabel_bns[i])
            spacegroup_label_unified[i] = line[3]
            spacegroup_label_bns[i] = line[4]
            nlabelparts_og[i,:] = line[5:8]
            nlabel_og[i] = line[8]
            spacegroup_label_og[i] = line[9]
            # print(spacegroup_label_og[i])
            magtype = int(file.readline().strip())
            if magtype == 4:
                matrix_data = file.readline().split()
                # print('matrix_data is,',matrix_data)
                bnsog_point_op[i,:,:] =  np.array(matrix_data[0:9], dtype=int).reshape(3, 3)
                # print("bnsog_point_op",bnsog_point_op[i,:,:])
                bnsog_origin[i,:] = matrix_data[9:12]
                bnsog_origin_denom[i] = matrix_data[12]

            ops_count[i] = int(file.readline().strip())
            # print('ops_count[i] is', ops_count[i])

            lines_data = []

            ops_lines = ((ops_count[i]-1)//4) + 1
            for num_line in range(ops_lines):
                line = file.readline().split()
                lines_data.append(line)

            # print('lines_data is', lines_data )

            for j in range(ops_count[i]): 
                ops_bns_point_op[i,j] = lines_data[j//4][0+6*(j%4)]
                ops_bns_trans[i,j,:] = lines_data[j//4][1+6*(j%4):4+6*(j%4)]
                ops_bns_trans_denom[i,j] = lines_data[j//4][4+6*(j%4)]
                ops_bns_timeinv[i,j] = lines_data[j//4][5+6*(j%4)]
                # print(ops_bns_point_op[i])
            lattice_bns_vectors_count[i] = int(file.readline().strip())
            # print("lattice_bns_vectors_count[i] ",lattice_bns_vectors_count[i])
            line = file.readline().split()

            # print(line)

            for j in range(lattice_bns_vectors_count[i]):    
                lattice_bns_vectors[i,j,:] = line[0+4*j:3+4*j]
                lattice_bns_vectors_denom[i,j] = line[3+4*j]

            wyckoff_site_count[i] =  int(file.readline().strip())
            # print('wyckoff_site_count[i] is', wyckoff_site_count[i])

            for j in range(wyckoff_site_count[i]):
                line = file.readline()
                matches = re.findall(r'[^"\s]\S*|"(?:\\.|[^"\\])*"', line)
                processed_matches = [match.strip('"') for match in matches]
                line = processed_matches
                # print("modified wyckoff label is", line)
                wyckoff_pos_count[i,j],wyckoff_mult[i,j],wyckoff_label[i,j] = line
                # print("wyckoff_label is", wyckoff_label[i,j])
                for k in range(wyckoff_pos_count[i,j]):
                    line = file.readline().split()
                    wyckoff_bns_fract[i,j,k,:] = line[0:3]
                    wyckoff_bns_fract_denom[i,j,k] = line[3]
                    wyckoff_bns_xyz[i,j,k,:,:] = np.array(line[4:13], dtype=int).reshape(3, 3).transpose()
                    wyckoff_bns_mag[i,j,k,:,:] = np.array(line[13:22], dtype=int).reshape(3, 3).transpose()
                    # print("wyckoff_bns_xyz is",wyckoff_bns_xyz[i,j,k,:,:])

            if magtype == 4:           
                ops_count[i] = int(file.readline().strip())
                # print('ops_count[i] is', ops_count[i])

                lines_data = []
                # if ops_count[i] <= 4:
                #     line = file.readline().split()
                #     for j in range(ops_count[i]): 
                #         ops_og_point_op[i,j] = line[0+6*j]
                #         ops_og_trans[i,j,:] = line[1+6*j:4+6*j]
                #         ops_og_trans_denom[i,j] = line[4+6*j]
                #         ops_og_timeinv[i,j] = line[5+6*j]
                # else:
                ops_lines = (ops_count[i]-1)//4 + 1
                for num_line in range(ops_lines):
                    line = file.readline().split()
                    lines_data.append(line)

                # print('lines_data is', lines_data )

                for j in range(ops_count[i]): 
                    #ops_og_point_op[i,j] = lines_data[(ops_count[i]-1)//4][0+6*(j%4)]   ####  wrong
                    ops_og_point_op[i,j] = lines_data[j//4][0+6*(j%4)]
                    # print(ops_og_point_op[i])
                    ops_og_trans[i,j,:] = lines_data[j//4][1+6*(j%4):4+6*(j%4)]
                    ops_og_trans_denom[i,j] = lines_data[j//4][4+6*(j%4)]
                    ops_og_timeinv[i,j] = lines_data[j//4][5+6*(j%4)]

                lattice_og_vectors_count[i] = int(file.readline().strip())

                line = file.readline().split()
                for j in range(lattice_og_vectors_count[i]):    
                    lattice_og_vectors[i,j,:] = line[0+4*j:3+4*j]
                    lattice_og_vectors_denom[i,j] = line[3+4*j]


                wyckoff_site_count[i] =  int(file.readline().strip())

                for j in range(wyckoff_site_count[i]):

                    line = file.readline()
                    matches = re.findall(r'[^"\s]\S*|"(?:\\.|[^"\\])*"', line)
                    processed_matches = [match.strip('"') for match in matches]
                    line = processed_matches
                    # print("modified wyckoff label is", line)
                    wyckoff_pos_count[i,j],wyckoff_mult[i,j],wyckoff_label[i,j] = line
                    # print("wyckoff_label is", wyckoff_label[i,j])

                    for k in range(wyckoff_pos_count[i,j]):
                        line = file.readline().split()
                        wyckoff_og_fract[i,j,k,:] = line[0:3]
                        wyckoff_og_fract_denom[i,j,k] = line[3]
                        wyckoff_og_xyz[i,j,k,:,:] =  np.array(line[4:13], dtype=int).reshape(3, 3).transpose()
                        wyckoff_og_mag[i,j,k,:,:] =  np.array(line[13:22], dtype=int).reshape(3, 3).transpose()
                        # print("wyckoff_og_xyz is",wyckoff_og_xyz[i,j,k,:,:])

# def bns_symm_dictionary():
#     for i in range(1651):
#         if int(nlabelparts_bns[i,0]) >= 168 and int(nlabelparts_bns[i,0]) <= 194:
#             for j in range(ops_count[i]):
#                 print(int(ops_bns_point_op[i,j]))
#                 op_label = point_op_hex_label[int(ops_bns_point_op[i,j])-1]
#                 print(op_label)
#                 mag_symmety_data_bns[nlabelparts_og[i,2]]['operators_label'] = op_label
#         else:
#             for j in range(ops_count[i]):
#                 op_label =point_op_label[(ops_bns_point_op[i,j])-1]
#                 mag_symmety_data_bns[nlabelparts_og[i,2]]['operators_label'][j] =op_label
                
#         for j in range(ops_count[i]):
#             mag_symmety_data_bns[nlabelparts_og[i,2]]['operators_trans'][j] = ops_bns_trans[i,j,:]/int(ops_bns_trans_denom[i,j])
#             mag_symmety_data_bns[nlabelparts_og[i,2]]['ops_bns_timeinv'][j] = ops_bns_timeinv[i,j]
             
# def bns_symm_dictionary():
#     for i in range(1651):
#         key = nlabelparts_og[i, 2]
#         print(key)
#         # 确保每个键都初始化为包含空列表的字典
#         if key not in mag_symmety_data_bns:
#             mag_symmety_data_bns[key] = {'operators_label': [], 'operators_trans': [], 'ops_bns_timeinv': []}
#             # print(mag_symmety_data_bns)
#         if int(nlabelparts_bns[i, 0]) >= 168 and int(nlabelparts_bns[i, 0]) <= 194:
#             for j in range(ops_count[i]):
#                 op_label = point_op_hex_label[int(ops_bns_point_op[i, j]) - 1]
#                 # 对于特定条件下的operators_label，添加到列表
#                 mag_symmety_data_bns[key]['operators_label'].append(op_label)
#         else:
#             for j in range(ops_count[i]):
#                 op_label = point_op_label[int(ops_bns_point_op[i, j]) - 1]
#                 # 对于非特定条件下的operators_label，添加到列表
#                 mag_symmety_data_bns[key]['operators_label'].append(op_label)
        
#         # 对于所有的operators_trans和ops_bns_timeinv，添加到列表
#         for j in range(ops_count[i]):
#             mag_symmety_data_bns[key]['operators_trans'].append(ops_bns_trans[i, j, :] / int(ops_bns_trans_denom[i, j]))
#             mag_symmety_data_bns[key]['ops_bns_timeinv'].append(ops_bns_timeinv[i, j])

def bns_symm_dictionary():
    for i in range(1651):
        key = nlabelparts_og[i, 2]
        
        # 初始化一个临时列表来存储operators_label的值
        temp_operators_label = []
        
        for j in range(ops_count[i]):
            if int(nlabelparts_bns[i, 0]) >= 168 and int(nlabelparts_bns[i, 0]) <= 194:
                op_label = point_op_hex_label[int(ops_bns_point_op[i, j]) - 1]
            else:
                op_label = point_op_label[int(ops_bns_point_op[i, j]) - 1]
            
            # 添加到临时列表而不是字典
            temp_operators_label.append(op_label)
        
        # 现在为每个i的键赋值一次，包含了ops_count[i]个元素的列表
        if key not in mag_symmety_data_bns:
            mag_symmety_data_bns[key] = {
                'operators_label': [], 
                'operators_trans': [], 
                'ops_bns_timeinv': [], 
                'wyckoffs': []
                }
            
        
        
        # 赋值给字典
        mag_symmety_data_bns[key]['operators_label'] = temp_operators_label
        
        # 对于其他属性的处理，确保也是在循环外完成的
        temp_operators_trans = []
        temp_ops_bns_timeinv = []
        for j in range(ops_count[i]):
            temp_operators_trans.append(ops_bns_trans[i, j, :] / int(ops_bns_trans_denom[i, j]))
            temp_ops_bns_timeinv.append(ops_bns_timeinv[i, j])
        
        mag_symmety_data_bns[key]['operators_trans'] = temp_operators_trans
        mag_symmety_data_bns[key]['ops_bns_timeinv'] = temp_ops_bns_timeinv
      
        wyckoffs_list = []
        
        for j in range(wyckoff_site_count[i]):
            # wyckoff_info_site={
            # 'wyckoff_pos_count': []
            # }
            
            # wyckoff_pos_count['wyckoff_pos_count'].append(wyckoff_pos_count[i,j])
            wyckoff_info = {
                'wyckoff_mult': [],
                'wyckoff_label': [],
                'wyckoff_bns_fract' : [],
                'wyckoff_bns_fract_denom' : [],
                'wyckoff_bns_xyz' : [],
                'wyckoff_bns_mag' : [],
                'fraction_xyz_shift': []
            }
            
            for k in range(wyckoff_pos_count[i, j]):  # 假设coordinate_count[i, j]是每个位点坐标的数量
                
                coordinate_xyz = wyckoff_bns_xyz[i, j, k]  # 获取坐标的示例方法
                coordinate_mag = wyckoff_bns_mag[i, j, k]
                bns_xyz_shift =  wyckoff_bns_fract[i,j,k] / wyckoff_bns_fract_denom[i,j,k]
                fraction_values = str([nsimplify(value, tolerance=0.01) for value in bns_xyz_shift])
                
                
                x, y, z = symbols('x y z')
                sympy_matrix = Matrix(coordinate_xyz)
                vector = Matrix([x, y, z])
                result = sympy_matrix * vector
                result_str = ",".join([str(elem) for elem in result])
                # print(result_str)
                # wyckoff_info_pos['wyckoff_bns_xyz'].append(result_str)
                
                mx, my, mz = symbols('mx my mz')
                sympy_matrix = Matrix(coordinate_mag)
                vector = Matrix([mx, my, mz])
                result_mag = sympy_matrix * vector
                result_mag_str = ",".join([str(elem) for elem in result_mag])
                # print(result_mag_str)
                # wyckoff_info_pos['wyckoff_bns_mag'].append(result_mag_str)
                
                variables = result_str.split(",")
                fractions = fraction_values.split(",")

                # 更精确的拼接字符串，直接处理正负号和避免不必要的括号
                result_shift_str = ','.join([f"{var}+{frac}" if frac != '0' else var for var, frac in zip(variables, fractions)])
                result_shift_str = result_shift_str.replace('+-', '-')
                
                
                wyckoff_info['wyckoff_bns_xyz'].append(result_str)
                wyckoff_info['wyckoff_bns_mag'].append(result_mag_str)
                wyckoff_info['fraction_xyz_shift'].append(result_shift_str)
                
                # temp_wyckoff_mult = []
                # temp_wyckoff_mult = wyckoff_mult[i,j]
                # temp_wyckoff_label = []
                # temp_wyckoff_label= wyckoff_label[i,j]
                
                # wyckoff_info['wyckoff_mult'].append(temp_wyckoff_mult[k])
                # wyckoff_info['wyckoff_label'].append(temp_wyckoff_label[k])
                
                
                # print(wyckoff_point_info['wyckoff_bns_xyz'])
            wyckoffs_list.append(wyckoff_info)

        mag_symmety_data_bns[key]['wyckoffs']= wyckoffs_list

def gen_mag_table():
    with open(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\gen_og_table.txt', 'w') as file:
        for i in range(1651):        
            # for j in ops_count[i]:
            #     mag_symmety_data_og[nlabelparts_og[i,2]]['operators'] =ops_og_point_op[i,j]
            # mag_symmety_data_og[nlabelparts_og[i,2]] = {'operators' : for j in ops_count[i] : }
            # nlabelparts_og
            # print(ops_og_point_op[i])
            # np.savetxt(file, 'BNS ', nlabelparts_bns[i], 'OG ', nlabelparts_og[i], fmt='%s')
            label_bns_str = ' '.join(map(str, nlabelparts_bns[i]))
            label_og_str = ' '.join(map(str, nlabelparts_og[i]))
            spacegroup_label_unified_str = ' '.join(map(str,spacegroup_label_unified[i]))
            # 组合最终的字符串
            line = f"BNS {label_bns_str} OG {label_og_str} UNI_label {spacegroup_label_unified_str}\n"
            # 写入文件
            file.write(line)
            # ops_count_str = str(ops_bns_point_op[i])
            # line  = f"{ops_count_str}\n"
            # file.write(line)

            
            
            for j in range(ops_count[i]):
                
                op_label = str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['operators_label'][j])
                op_trans = str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['operators_trans'][j])
                op_timeinv = str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['ops_bns_timeinv'][j])
                line = f"'operators_label':{op_label},   op_trans:{op_trans} time_inv: {op_timeinv}\n"

            # 写入文件
                file.write(line)
            print(str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs']))
            for j in range(wyckoff_site_count[i]):
                # for k in range(wyckoff_pos_count[i,j]):
                #     print(k)
                #     # print(str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs']['wyckoff_bns_xyz'][k]))
                line = str(wyckoff_pos_count[i,j]) + ' ' \
                        + str(wyckoff_mult[i,j]) + ' ' \
                        + str(wyckoff_label[i,j]) + ' ' \
                        + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['fraction_xyz_shift'])+ ' '  \
                        + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_bns_mag'])+   '\n'
                        
                        # + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_mult']) + ' ' \
                        # + str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs'][j]['wyckoff_label'])+ ' ' \
                            

                file.write(line)
            # for j1 in range(wyckoff_site_count[i]):
                
            #     print(j1+1)
                # wyckoff_xyz = str(mag_symmety_data_bns[nlabelparts_og[i, 2]]['wyckoffs']['wyckoff_bns_xyz'])
                # print(wyckoff_xyz)
                # line = f"wyck_off_xyz:{wyckoff_xyz}\n"
                # file.write(line)
            # for j in range(ops_count[i]):
            #     line = f"mag_symmety_data_bns[nlabelparts_og[i,2]]['operators_label'][j]\n"
            #     file.write(line)
            # if int(nlabelparts_og[i,0]) >= 168 and int(nlabelparts_og[i,0]) <= 194:
            #     # print(ops_count[i])
            #     # print(ops_og_point_op[i])
            #     # print(point_op_hex_label[int(ops_og_point_op[i])])
            #     file.write(str(nlabelparts_bns[i,0]))
            #     file.write(str(ops_count[i]))
            #     file.write('\n')
                # ops_og_point_op[]
                
                
read_mag_txt() 
bns_symm_dictionary()

# print(bns_symm_dictionary)      
gen_mag_table()
print(wyckoff_site_count)
#print(nlabelparts_bns[1466])
# np.savetxt(r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\example.txt', wyckoff_og_xyz,fmt='%d')