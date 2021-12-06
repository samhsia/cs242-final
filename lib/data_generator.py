import os
import copy
import torch
import numpy as np
import multiprocessing

from lib.utils import *
from lib.approximation import Approximation
from lib.compute import Compute
from lib.experiments import Experiments


class ExecutionThread(multiprocessing.Process):
    def __init__(self, my_id, r_queue, core_queue, DATA_POINTS, DIMENSION, LUT_BITS, QUANTIZE_FACE, MAX_VAL, dtype):
        multiprocessing.Process.__init__(self)
        self.id = my_id
        self.r_queue = r_queue
        self.core_queue = core_queue
        self.DATA_POINTS = DATA_POINTS
        self.DIMENSION = DIMENSION
        self.LUT_BITS = LUT_BITS
        self.QUANTIZE_FACE = QUANTIZE_FACE
        self.MAX_VAL = MAX_VAL
        self.dtype = dtype

    def run(self):
        allowed_cores = self.core_queue.get()
        print("Experiment %s Started - Datapoints (%s) Weight Size (%s)" % (str(self.id), str(self.DATA_POINTS), str(self.DIMENSION)))
        os.sched_setaffinity(0, { allowed_cores[0], allowed_cores[1], allowed_cores[2] })

        approx = Approximation(self.MAX_VAL, self.dtype)
        comp = Compute(self.dtype)
        exp = Experiments(approx, comp, self.dtype)

        a_orig, a_conv, b_orig = initialize_data(self.DATA_POINTS, self.DIMENSION, self.LUT_BITS, self.QUANTIZE_FACE, approx)
        data = exp.sweep_approximation_pool(a_orig, a_conv, b_orig, STEP=int(self.DIMENSION/16), compute_method="blas")
        data["id"] = self.id
        self.r_queue.put(data)
        print("Experiment %s Completed" % self.id)
        self.core_queue.put(allowed_cores)
        return 

def create_csv_data_sweep(LUT_BITS, QUANTIZE_FACE, MAX_VAL, dtype, compute_method="blas", OVERWRITE=False):
    name = "crystal_skull_sweep.csv"
    
    header = ["id", "data-points", "dimension", "compute-only", "approx-only", "hybrid-best-exe", "hybrid-best-percent", "percent-approximated", "exec-time-compute", "exec-time-approx"]
    all_data = {}

    batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ] # 2048, 4096, 8192, 16384
    weight_size = [16, 32, 64, 128, 256, 512]

    r_queue = multiprocessing.Queue()
    core_queue = multiprocessing.Queue()
    for i in range(3, 12, 3):
        core_queue.put([i, i+1, i+2])


    id_old_max = 0

    if OVERWRITE:
        f = open(name, "w")
        f.write(", ".join(header) + "\n")
        f.close()
    else:  
        f = open(name, "r")
        all_lines = f.readlines()
        f.close()
        old_data = {}
        old_header = all_lines[0].replace("\n", "").split(", ")
        for line in all_lines[1:]:
            line = line.replace("\n", "").split(", ")
            old_line = {}
            for i, h in enumerate(old_header):
                old_line[h] = line[i]
            if int(old_line["id"]) > id_old_max:
                id_old_max = int(old_line["id"])
            old_data[str(old_line["data-points"]) + "-" + str(old_line["dimension"])] = old_line
        

    threads = []
    my_id_start = 0 if OVERWRITE else id_old_max+1
    my_id = my_id_start
    for DATA_POINTS in batch_size:
        for DIMENSION in weight_size:                    
            threads.append(ExecutionThread(my_id, r_queue, core_queue, DATA_POINTS, DIMENSION, LUT_BITS, QUANTIZE_FACE, MAX_VAL, dtype))
            my_id += 1

    for t in threads:
        t.start()
    
    for t in threads:
        data = r_queue.get()
        new_row = []
        for h in header:
            if h == "data-points":
                new_row.append(str(data["a-shape"][0]))
            elif h == "dimension":
                new_row.append(str(data["b-shape"][0]))
            else:
                new_row.append(str(data[h]).replace(",", ""))
        f = open(name, "a")
        f.write(", ".join(new_row) + "\n")
        f.close()
        
        threads[data["id"]-(my_id_start)].join()
