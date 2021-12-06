import time
import copy
import numpy as np
import multiprocessing
from operator import itemgetter


class Approximation:
    def __init__(self, MAX_VAL):
        self.MAX_VAL = MAX_VAL
        self.lut = None
        self.vals = None

    def create_lut(self, W, NUM_BITS):
        interval = (self.MAX_VAL * 2) / (2**NUM_BITS)
        vals = list(np.arange(-self.MAX_VAL, self.MAX_VAL, interval))

        T = {}
        for i, row in enumerate(W):
            if i not in T:
                T[i] = {}
            for j, val in enumerate(vals):
                if j not in T[i]:
                    T[i][j] = np.zeros(np.shape(W)[1], dtype=np.float16)
                    #T[i][j] = [0] * np.shape(W)[1]
                T[i][j] += val * row

        self.lut = T
        self.vals = vals

    def convert_data(self, data, n_cols):
        for row in range(np.shape(data)[0]):
            for col in range(n_cols):
                data[row][col] = self.get_bin(data[row][col])
        return data

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

    def cluster_inference_parallel(self, id, thread_lock, final_answer_lock, shared_lut, shared_vals, return_data,
                                   a_shared, b_shared, a_cols, b_rows):
        return ClusterInferenceThread(id, thread_lock, final_answer_lock, shared_lut, shared_vals, return_data,
                                      a_shared, b_shared, a_cols, b_rows, self.MAX_VAL)

class ClusterInferenceThread(multiprocessing.Process):
    def __init__(self, id, thread_lock, final_answer_lock, shared_lut, shared_vals, return_data,
                 a_shared, b_shared, a_cols, b_rows, MAX_VAL):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.thread_lock = thread_lock
        self.final_answer_lock = final_answer_lock

        self.a_shared = a_shared
        self.b_shared = b_shared
        self.shared_lut = shared_lut
        self.shared_vals = shared_vals
        self.return_data = return_data

        self.a_cols = a_cols
        self.b_rows = b_rows
        self.my_lut = {}
        for b_row_index in self.b_rows:
            self.my_lut[b_row_index] = self.shared_lut[b_row_index]

        self.b_shape = (len(b_rows), int(np.shape(list(self.shared_lut[0].values()))[1]))
        self.MAX_VAL = MAX_VAL

    def run(self):
        exe_time = 0
        final = np.zeros((np.shape(self.a_shared)[1], self.b_shape[1]))

        self.thread_lock.acquire()
        t_start = time.perf_counter()

        for compute_face in range(len(self.b_rows)):
            t_intermediate = self.my_lut[self.b_rows[compute_face]]
            bins = self.a_shared[self.a_cols[compute_face]]
            bins = tuple(bins)

            t1 = time.perf_counter()
            lookup = itemgetter(*bins)(t_intermediate)
            t2 = time.perf_counter()
            final += lookup
            exe_time += t2 - t1

        t_end = time.perf_counter()
        self.thread_lock.release()

        '''
        self.final_answer_lock.acquire()
        self.return_data["face"] += final
        if self.return_data["exe_time_approx"] < exe_time:
            self.return_data["exe_time_approx"] = exe_time
        self.final_answer_lock.release()
        '''
