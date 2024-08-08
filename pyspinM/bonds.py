import numpy as np
from pymatgen.core import Structure, Lattice
from scipy.linalg import null_space, orth
from read_cif import parse_and_symmetrize_structure
class Bond:
    def __init__(self, idx, subidx, dl, dr, length, matom1, idx1, matom2, idx2, matrices):
        self.idx = idx          # 按键长分的序号
        self.subidx = subidx    # 所有idx相同的键按对称性分组后的指标
        self.dl = dl            # 键箭头矢量指向的末端原子所在的胞的指标
        self.dr = dr            # 从末端原子看需要平移的晶格矢量倍数
        self.length = length    # 键长
        self.matom1 = matom1    # 初始端原子的类型
        self.idx1 = idx1        # 初始端原子在000原胞内的编号
        self.matom2 = matom2    # 末端原子类型
        self.idx2 = idx2        # 末端原子在相应平移后单胞里的编号        
        self.matrices = {
            'heisenberg': np.zeros((3, 3)),
            'dmi': np.zeros((3, 3)),
            'anisotropy': np.zeros((3, 3))
        }

    def set_matrix(self, matrix_type, matrix_values):
        """ 设置特定类型的矩阵 """
        if matrix_type in self.matrices and isinstance(matrix_values, np.ndarray) and matrix_values.shape == (3, 3):
            self.matrices[matrix_type] = matrix_values
        else:
            raise ValueError("Invalid matrix type or matrix shape. Matrix must be a 3x3 NumPy array.")

    def get_matrix(self, matrix_type):
        """ 获取特定类型的矩阵 """
        return self.matrices.get(matrix_type, None)

# def gencoupling(structure):

# for i, neighbors in enumerate(all_neighbors):
#     for neighbor in neighbors:
#         j = neighbor.index
#         dist = neighbor.nn_distance
#         # 暂时假设 idx, subidx 等其他值
#         idx = 1  # 这些值需要根据您的具体需求来定义
#         subidx = 0
#         dl = tuple(structure.lattice.get_fractional_coords(neighbor.coords))
#         dr = tuple(-structure[i].frac_coords + neighbor.frac_coords)  # 计算相对位移
#         length = dist
#         matom1 = structure[i].species_string
#         idx1 = i
#         matom2 = structure[j].species_string
#         idx2 = j
#         matrix = {'heisenberg': np.zeros((3, 3)), 'dmi': np.zeros((3, 3)), 'anisotropy': np.zeros((3, 3))}

#         bond = Bond(idx, subidx, dl, dr, length, matom1, idx1, matom2, idx2, matrix)
#         bonds.append(bond)

# 现在 bonds 列表包含了所有的 Bond 对象

def intersec(U1, U2):
    """
    返回两个向量空间 U1 和 U2 的交集的基。
    每个向量空间的基向量是输入矩阵的列。
    """
    # 计算 [U1 U2] 的零空间
    N = null_space(np.hstack([U1, U2]))
    # 计算 U1 和 U2 的交集的基
    I = U1 @ N[:U1.shape[1], :]
    # 返回正交且尺寸最小的基
    return orth(I)

import numpy as np
from scipy.linalg import null_space, orth


def indep(M, v, tol):
    """
    返回无法仅通过 M 的列向量线性组合得到的向量 v 的列向量。
    参数:
    M : numpy.ndarray
        已存在的列向量集合。
    v : numpy.ndarray
        需要检查的列向量集合。
    tol : float
        用于判断秩的容差。

    返回:
    numpy.ndarray
        从 v 中返回无法由 M 的列线性组合得到的列向量。
    """

    rank_M = np.linalg.matrix_rank(M, tol=tol)

    independent_indices = []

    for idx in range(v.shape[1]):

        augmented_matrix = np.hstack([M, v[:, [idx]]])
        rank_augmented = np.linalg.matrix_rank(augmented_matrix, tol=tol)
        
        if rank_augmented > rank_M:
            independent_indices.append(idx)

    return v[:, independent_indices]


def oporder(symOp):
    """
    计算旋转操作的阶。
    symOp 应该是一个 4x4 矩阵，其中前 3x3 是旋转矩阵 RN，最后一列是平移向量。
    """
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
    nSym = 2
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

    print("M0_V",M0_V)
    norm_r = np.linalg.norm(r)
    if norm_r > 0:
        r = r / norm_r
        aniso = False
    else:
        aniso = True

    for ii in range(nSym):
        
        R = symOp[:, :, ii]
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
        I9 = np.eye(9)
        kron_R_R = np.kron(R, R)
        U, d, MS = np.linalg.svd(kron_R_R - I9)
        D = np.zeros((U.shape[1], MS.shape[0]), dtype=np.float64)
        np.fill_diagonal(D, d)   
        M_S = MS.T[:, np.abs(np.diag(D)) < tol].reshape(3, 3, -1)
        # 
        print("M_S is", M_S)
        print("MS is", MS)
        if parR == -1:
            U, d, MA = np.linalg.svd(kron_R_R + I9)
            D = np.zeros((U.shape[1], MA.shape[0]), dtype=np.float64)
            np.fill_diagonal(D, d)
            print("MA is\n",MA)

            M_A = MA.T[:, np.abs(np.diag(D)) < tol].reshape(3, 3, -1)
            M_A = M_A - np.transpose(M_A, (1, 0, 2))
            
        else:
            M_A = M_S - np.transpose(M_S, (1, 0, 2))

        M_S = M_S + np.transpose(M_S, (1, 0, 2))
        M_V = np.reshape(np.concatenate((M_S, M_A), axis=2), (9, -1))
        M0_V = intersec(M0_V, M_V) 
        
        normM = np.array([np.linalg.norm(M0_V[:, idx]) for idx in range(M0_V.shape[1])])
        M0_V = M0_V[:, normM >= tol]

    M0_V = M0_V.reshape(3, 3, -1)
    symmetric = M0_V + np.transpose(M0_V, (1, 0, 2))  # 计算对称部分
    antisymmetric = M0_V - np.transpose(M0_V, (1, 0, 2))  # 计算反对称部分
    M0_V = np.concatenate((antisymmetric, symmetric), axis=2).reshape(9, -1)  # 重塑回 (9, num_matrices*2)
    
    normM = np.linalg.norm(M0_V, axis=0)
    M0_V = M0_V[:, normM >= tol]

    rank_M0_V = np.linalg.matrix_rank(M0_V, tol=tol)
    
    rM = np.array([np.linalg.matrix_rank(np.hstack([M0_V, V0[:, [idx]]]), tol=tol) for idx in range(V0.shape[1])])

    Vnice = V0[:, rM == rank_M0_V]

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
    M  = M.reshape(3, 3, -1)

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

    asym = np.concatenate([asym[~asym], asym[asym]])

    return M,asym
    
dl = np.array([0,0,0])
# dr = np.array([-1,-1,0])
center = np.array([0,-0.5,0.])
pOp = np.zeros((3, 3, 6))

pOp[:,:,0] = np.array( [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
pOp[:,:,1] = np.array([ [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
pOp[:,:,2] = np.array( [[ 0, -1,  0],
                        [ 1, -1,  0],
                        [ 0,  0,  1]])
pOp[:,:,3] = np.array( [[ 0,  1,  0],
                        [-1,  1,  0],
                        [ 0,  0, -1]])
pOp[:,:,4] = np.array( [[-1,  1,  0],
                        [-1,  0,  0],
                        [ 0,  0,  1]])
pOp[:,:,5] = np.array(  [[ 1, -1,  0],
                        [ 1,  0,  0],
                        [ 0,  0, -1]])

def point_Op(rotations,trans,r):
    is_point_op = np.full(len(rotations), False, dtype=bool)
    for i, rot in enumerate(rotations):
        new_r = np.dot(rot,r) % 1
        if np.allclose(new_r, r, atol=1e-5):
            is_point_op[i] = True
    pOp = np.zeros((3, 3, len(rotations)))
    for j in range(len(rotations)):
        pOp[:,:,j] = rotations[j]
    return pOp
    # return rotations[is_point_op]


# A = np.array([[6, 0, 0],
#               [0, 4, 0],
#               [0, 0, 4]])
# cif_file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\VO2P-4m2.cif'
cif_file_path = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\P3_cell.cif'
# cif_file_path2 = r'C:\Users\wangh\OneDrive\Desktop\Codes\SymmMagHam\pyspinM\CrI3.cif'
symmetry_dataset, result_structure = parse_and_symmetrize_structure(cif_file_path)

# A = result_structure.lattice

test_vector = np.array([0.25,0.,0.4765])

point_rot = point_Op(symmetry_dataset['rotations'],symmetry_dataset['translations'],test_vector)

lattice_matrix = result_structure.lattice
# dr =
print("lattice_matrix is ",lattice_matrix)
A = np.array([[ 6.000000 ,0.000000, 0.000000],
              [-3.000000, 5.196152, 0.000000],
              [0.000000, 0.000000, 14.000000]])

product = A @ dr
print(product)

aMat, aSym = basic_sym_matrix(point_rot, product,1e-5)

print("aMat:", aMat)
print("aSym:", aSym)

aMatS = aMat[:, :, ~aSym]
print(aMatS)
for i in range(aMatS.shape[2]):
    print(f"Matrix {i+1}:\n{aMatS[:, :, i]}\n")
    
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

tol = 1e-5

import numpy as np
nargout = 0
if nargout == 0:
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
                    eStr[jj][kk] += f"{aMatS[jj, kk, ii]:.2f}{chr(65 + ii)}"

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
# print("point op for ",test_vector,"are \n")