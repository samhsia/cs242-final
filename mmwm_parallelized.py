import os
import copy
import torch
import numpy as np
import multiprocessing
np.random.seed(42)

from lib.utils import *
from lib.data_generator import *
from lib.approximation import Approximation
from lib.compute import Compute
from lib.experiments import Experiments

# Matrix Initilalization
DATA_POINTS = 256       # 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
DIMENSION = 64         # 64, 128, 256, 512, 1024, 2048, 4096

# Approximation Hyperparameters
MAX_VAL = 4
LUT_BITS = 8
QUANTIZE_FACE = True

dtype=np.float16

#initialize_all(MAX_VAL, LUT_BITS, QUANTIZE_FACE, dtype)

if __name__ == "__main__":
    os.system("clear")
    print("################################################################")
    multiprocessing.set_start_method('fork')

    approx = Approximation(MAX_VAL, dtype)
    comp = Compute(dtype)
    exp = Experiments(approx, comp, dtype)

    # Note: if you are modeling with new matrices, set reinitialize to True
    a_orig, a_conv, b_orig = initialize_data(DATA_POINTS, DIMENSION, LUT_BITS, QUANTIZE_FACE, approx, reinitialize=False)
    
    # run a single experiment for MMM!
    exp.sweep_approximation_pool(a_orig, a_conv, b_orig, STEP=int(DIMENSION/16), compute_method="blas") # compute_method either "outer" or "blas"
    
    
    # use this for a sweep to generate a csv of data
    # warning - creates lots of processes and can take a while to run in incorrectly initialized!
    #create_csv_data_sweep(LUT_BITS, QUANTIZE_FACE, MAX_VAL, dtype, compute_method="blas")
    
    print("################################################################")


