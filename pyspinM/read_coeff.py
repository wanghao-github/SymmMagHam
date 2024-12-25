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
import json


def collect_energy_OUTCAR(directory="."):
    outcar_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "OUTCAR":
                outcar_files.append(os.path.join(root,file))
    return outcar_files

def read_coeff_json(base_dir,):
    