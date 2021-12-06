import os
import sys
import time
import copy
import numpy as np
import multiprocessing
from operator import itemgetter


class Approximation:
    def __init__(self, MAX_VAL, dtype):
        self.MAX_VAL = MAX_VAL
        self.lut = None
        self.vals = None
        self.dtype = dtype
        self.NUM_BITS = None
        self.QUANTIZE_FACE = None
        self.QUANTIZED_TARGET = 16

    def dequantize_row(self, row): 
        max_val = self.MAX_VAL**2
        return (((row + (2**self.QUANTIZED_TARGET)/2) / (2**self.QUANTIZED_TARGET)) * (2*(max_val)) - max_val).astype(self.dtype)

    def quantize_row(self, row):
        new_row = np.zeros(np.shape(row), dtype=int)
        max_val = self.MAX_VAL**2  # needed to be squared because max of row is self.MAX_VAL which is multiplied by val which has a max of self.MAX_VAL
        
        row[np.where(row > max_val)] = max_val # clamp top 
        row[np.where(row < -1*max_val)] = -1*max_val # clamp bottom
        new_row += (((row + max_val) / (max_val * 2) * (2**self.QUANTIZED_TARGET)) - (2**self.QUANTIZED_TARGET)/2).astype(int)
        return new_row
    
    def row_table(self, r_queue, i, row, vals, QUANTIZE_FACE):
        T_row = {}
        for j, val in enumerate(vals):
            if QUANTIZE_FACE == False:
                T_row[j] = np.array(val * row, dtype=self.dtype)
            else:
                T_row[j] = self.quantize_row(val * row)
        r_queue.put([i, T_row])
        return 
  
    def create_lut_parallel(self, W, NUM_BITS, QUANTIZE_FACE=False):
        self.QUANTIZE_FACE = QUANTIZE_FACE
        self.NUM_BITS = NUM_BITS
        interval = (self.MAX_VAL * 2) / (2**NUM_BITS)
        vals = list(np.arange(-self.MAX_VAL, self.MAX_VAL, interval))
        
        T = {}
        r_queue = multiprocessing.Queue()
        threads = []

        for i, row in enumerate(W):
            threads.append(multiprocessing.Process(target=self.row_table, args=(r_queue, i, row, vals, QUANTIZE_FACE)))
        
        for i, t in enumerate(threads):
            t.start()
            if i >= 128:
                val = r_queue.get()
                T[val[0]] = val[1]
                sys.stdout.write(f'\r\tPercent Completed: {i-128} / {len(threads)} - {int(100*(i-128)/len(threads))}%')
                sys.stdout.flush()
        
        for i in range(128):
            val = r_queue.get()
            T[val[0]] = val[1]
            sys.stdout.write(f'\r\tPercent Completed: {len(threads) - 128 + i} / {len(threads)} - {int(100*(len(threads) - 128 + i)/len(threads))}%')
            sys.stdout.flush()
        
        for t in threads:
            t.join()
        
        print("")

        self.lut = T
        self.vals = vals
    
    def create_lut(self, W, NUM_BITS, QUANTIZE_FACE=False):
        self.QUANTIZE_FACE = QUANTIZE_FACE
        self.NUM_BITS = NUM_BITS
        interval = (self.MAX_VAL * 2) / (2**NUM_BITS)
        vals = list(np.arange(-self.MAX_VAL, self.MAX_VAL, interval))
        
        T = {}
        for i, row in enumerate(W):
            if i not in T:
                T[i] = {}
            for j, val in enumerate(vals):
                if j not in T[i]:
                    if QUANTIZE_FACE == False:
                        T[i][j] = np.zeros(np.shape(W)[1], dtype=self.dtype) 
                if QUANTIZE_FACE == False:
                    T[i][j] += val * row
                else:
                    new_vals = val * row
                    T[i][j] = self.quantize_row(new_vals)
        self.lut = T
        self.vals = vals

    def load_lut(self, T, vals, NUM_BITS, QUANTIZE_FACE):
        self.lut = T
        self.vals = vals
        self.NUM_BITS = NUM_BITS
        self.QUANTIZE_FACE = QUANTIZE_FACE

    def convert_data(self, data):
        r_queue = multiprocessing.Queue()
        
        threads = []
        for row in range(np.shape(data)[0]):
            threads.append(multiprocessing.Process(target=self.get_bin_parallel_fast, args=(row, data[row], r_queue)))

        for i, t in enumerate(threads):
            t.start()
            if i >= 128:
                val = r_queue.get()
                data[val[0]] = val[1]
                sys.stdout.write(f'\r\tPercent Completed: {i-128} / {len(threads)} - {int(100*(i-128)/len(threads))}%')
                sys.stdout.flush()
        
        for i in range(128):
            val = r_queue.get()
            data[val[0]] = val[1]
            sys.stdout.write(f'\r\tPercent Completed: {i-128} / {len(threads)} - {int(100*(len(threads) - 128 + i)/len(threads))}%')
            sys.stdout.flush()

        for t in threads:
            t.join()
        print("")

        return data

    def get_bin_parallel_fast(self, row, row_data, r_queue):
        conv_data = np.zeros(len(row_data), dtype=int)
        for i, d_val in enumerate(row_data):
            conv_data[i] = int((np.abs(self.vals - d_val)).argmin())
        r_queue.put([row, conv_data])
        return

    def get_bin(self, d_val):
        for index, val in enumerate(self.vals):
            if d_val < val:
                return index if abs(d_val-self.vals[index]) < abs(d_val - self.vals[index-1]) else (index-1)
        return len(self.vals) - 1

    def cluster_inference(self, data_points):
        face = np.zeros((np.shape(data_points)[1], len(self.lut.keys())))
        exe_time = 0
        for weight_row, d in enumerate(data_points):
            t_intermediate = self.lut[weight_row]
            bins = []

            for d_val in d:
                d_val = self.MAX_VAL if d_val > self.MAX_VAL else d_val
                d_val = -self.MAX_VAL if d_val < -1*self.MAX_VAL else d_val

                bin = self.get_bin(d_val)
                bins.append(bin)

            bins = tuple(bins)

            t1 = time.perf_counter()
            lookup = itemgetter(*bins)(t_intermediate)
            t2 = time.perf_counter()
            exe_time += t2 - t1

            if weight_row == 0:
                t1 = time.perf_counter()
                face += lookup
                t2 = time.perf_counter()
                exe_time += t2 - t1
            else:
                face += lookup

        return face, exe_time/8

    def cluster_inference_parallel(self, x, barrier, queue, lock, r_queue):
        if len(list(os.sched_getaffinity(os.getpid()))) > 1:
            os.sched_setaffinity(0, { list(os.sched_getaffinity(os.getpid()))[0] })
            
        # parse input
        a = x[1]
        faces = x[2]

        # initialize variables
        exe_time = 0

        barrier.wait()

        for f in faces:
            t1 = time.perf_counter()
            bins = a[:, f]
            t2 = time.perf_counter()
            exe_time += t2-t1
            
            bins = tuple(bins.astype(int)) # slow! irrelevant to the problem so not timed
            t_intermediate = self.lut[f]
            
            t1 = time.perf_counter()
            lookup = itemgetter(*bins)(t_intermediate)
            t2 = time.perf_counter()

            queue.put(["Approx", lookup])
            exe_time += (t2-t1)

        queue.put(["DONE", None])
        exe_time = exe_time if len(faces) > 0 else 0

        r_queue.put(["Approx", exe_time, None])        
        return

