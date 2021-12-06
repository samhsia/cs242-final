import os
import sys
import time
import copy
import torch
import pickle
import numpy as np
from timeit import default_timer as timer  

from lib.approximation import Approximation


def cpu_exp(a_orig, b_orig, comp):
    t1 = time.perf_counter()
    c_orig = np.dot(a_orig, b_orig)
    t2 = time.perf_counter()
    print("CPU MATMUL %0.2fms" % ((t2-t1)*1000))

def gpu_exp(a_orig, b_orig, comp):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    a_gpu = torch.tensor(copy.deepcopy(a_orig)).to(device="cuda")
    b_gpu = torch.tensor(copy.deepcopy(b_orig)).to(device="cuda")
    
    torch.cuda.synchronize()
    start.record()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    c_gpu = c_gpu.to(device="cpu")
    end.record()

    torch.cuda.synchronize()
    print("GPU MATMUL %0.2fms" % (start.elapsed_time(end)))
    #comp.check_accuracy(a_orig, b_orig, np.array(c_gpu))

def initialize_data(DATA_POINTS, DIMENSION, LUT_BITS, QUANTIZE_FACE, approx, reinitialize=False):
    dtype=np.float16

    quant = "quantized" if QUANTIZE_FACE else "float"  
    name_a = "matrix_a_" + str(LUT_BITS) + "_" + quant + ".pkl"
    name_b = "matrix_b_" + str(DIMENSION)   + "_" + str(LUT_BITS) + "_" + quant + ".pkl"

    prefix = "./init/"

    # create random matrix B
    if not os.path.exists(prefix + name_b) or reinitialize:
        print("\tCreating Data Matrix B - " + name_b)
        b_orig = np.array(np.random.randn(DIMENSION, DIMENSION), dtype=dtype)

        # create the approximation object for matrix B and compute object
        approx.create_lut(b_orig, LUT_BITS, QUANTIZE_FACE)
        T = approx.lut
        vals = approx.vals
        
        data = {
                "b_orig" : b_orig, 
                "T"      : T,
                "vals"   : vals
               }
        with open(prefix + name_b, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("\tFinished Creating Data for B")
    else:
        #print("\tLoading Data for B - " + str(name_b))
        with open(prefix + name_b, "rb") as handle:
            data = pickle.load(handle)
            b_orig = data["b_orig"]
            T      = data["T"]
            vals   = data["vals"]

            approx.load_lut(T, vals, LUT_BITS, QUANTIZE_FACE)

    # create random matrix A
    if not os.path.exists(prefix + name_a) or reinitialize:
        print("\tCreating Data Matrix A - " + name_a)
        a_orig = np.array(np.random.randn(16384, 4096), dtype=dtype)
        a_conv = approx.convert_data(copy.deepcopy(a_orig))
        data = {
                    "a_orig" : a_orig, 
                    "a_conv" : a_conv
                 }

        with open(prefix + name_a, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("\tFinished Creating Data for A")
    else:
        #print("\tLoading Data for A - " + str(name_a))
        with open(prefix + name_a, "rb") as handle:
            data = pickle.load(handle)
            a_orig = data["a_orig"][0:DATA_POINTS, 0:DIMENSION]
            a_conv = data["a_conv"][0:DATA_POINTS, 0:DIMENSION]
    
    return a_orig, a_conv, b_orig

def initialize_all(MAX_VAL, LUT_BITS, QUANTIZE_FACE, dtype):
    for DATA_POINTS in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        for DIMENSION in [64, 128, 256, 512, 1024]:
            approx = Approximation(MAX_VAL, dtype)
            print("Data points=%i - Dimension=%i" % (DATA_POINTS, DIMENSION))
            a_orig, a_conv, b_orig = initialize_data(DATA_POINTS, DIMENSION, LUT_BITS, QUANTIZE_FACE, approx)

def best_performance(exec_time_approx_sweep, exec_time_compute_sweep, percent_approximated):
    best_exe_time = sys.maxsize 
    best_percent_approx = 0
    for i in range(len(percent_approximated)):
        curr_exe_time = max(exec_time_compute_sweep[i], exec_time_approx_sweep[i])
        if curr_exe_time < best_exe_time:
            best_exe_time = curr_exe_time
            best_percent_approx = percent_approximated[i]
    return best_exe_time, best_percent_approx