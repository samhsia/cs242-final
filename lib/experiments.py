import os
import sys
import copy
import numpy as np
import multiprocessing

from lib.utils import *
from lib.visualization import Visualization

class Experiments:
    def __init__(self, approx, compute, dtype):
        self.approx = approx
        self.compute = compute
        self.v = Visualization()
        self.dtype = dtype

    def _pool_init(self, l, b, q):
        global lock, barrier, queue
        lock, barrier, queue = l, b, q
        my_pid = os.getpid() # set cores this pool can send jobs too  
        #os.sched_setaffinity(0, {1, 3, 5}) # possibly need to set the 0 to my_pid

    def single_approximation(self, a_orig, b_orig, n_faces=0):
        a = copy.deepcopy(a_orig)
        b = copy.deepcopy(b_orig)

        c_approx_zero, exe_time_approx = self.approx.cluster_inference(a[:, 0:n_faces].T)
        c_compute, exe_time_compute = self.compute.outer(a[:, n_faces:], b[n_faces:])
        c_final = c_compute + c_approx_zero

        f_loss = self.compute.check_accuracy(a, b, c_final, False)

        return exe_time_approx, exe_time_compute, f_loss

    def create_pool_input(self, a, a_conv, b, n_faces, comp_m):
        x = [
             ("approx",  a_conv[:, 0:n_faces], list(range(0, n_faces))), # change this to the correct b values for better accuracy
             (comp_m,    a[:, n_faces:], b[n_faces:]),
             ("sum",     a, b, self.approx.QUANTIZE_FACE, self.approx)
            ]
        return x

    def single_approximation_parallel(self, x):
        if x[0] == "approx":
            return self.approx.cluster_inference_parallel(x, barrier, queue, lock)
        if x[0] == "outer":
            return self.compute.outer_parallel(x, barrier, queue, lock)
        if x[0] == "blas":
            return self.compute.blas_parallel(x, barrier, queue, lock)
        if x[0] == "sum":
            return self.compute.sum_parallel(x, barrier, queue, lock)

    def single_approximation_parallel_process(self, x, barrier, queue, lock, r_queue):
        if x[0] == "approx":
            return multiprocessing.Process(target=self.approx.cluster_inference_parallel, args=(x, barrier, queue, lock, r_queue))
        if x[0] == "outer":
            return multiprocessing.Process(target=self.compute.outer_parallel, args=(x, barrier, queue, lock, r_queue))
        if x[0] == "blas":
            return multiprocessing.Process(target=self.compute.blas_parallel, args=(x, barrier, queue, lock, r_queue))
        if x[0] == "blas-gpu":
            return multiprocessing.Process(target=self.compute.blas_gpu_parallel, args=(x, barrier, queue, lock, r_queue))
        if x[0] == "sum":
            return multiprocessing.Process(target=self.compute.sum_parallel, args=(x, barrier, queue, lock, r_queue))


    def sweep_approximation_pool(self, a_orig, a_conv, b_orig, STEP=1, compute_method="blas"):
        #####################################################################################
        # define constants
        TOTAL_FACES = np.shape(a_orig)[1]
        ALL_FACES = np.arange(0, TOTAL_FACES+1, STEP) # Try to change range

        a = a_orig # used to be deepcopy but no longer needed
        b = b_orig # used to be deepcopy but no longer needed

        # create datastructures for saving stats
        exec_time_approx_sweep = []
        exec_time_compute_sweep = []
        exec_time_sum_sweep = []
        f_loss_results = []
        svd_comparison = []
        percent_approximated = []

        results = []

        # shared datastructure for processing in parallel
        lock = multiprocessing.Lock()           # only used for testing... may remove later
        barrier = multiprocessing.Barrier(3)    # align start of compute process, approx process, sum process
        queue = multiprocessing.Queue()         # feed the sum process from compute or approx
        r_queue = multiprocessing.Queue()       # the return queue for each process
        #####################################################################################



        #####################################################################################
        # sweep the number of faces using a Pool
        for i, n_faces in enumerate(ALL_FACES):
            sys.stdout.write('\r\tStarting Experiments')
            sys.stdout.write('\r\tPercent Completed: {} / {} - {}%'.format(i+1, len(ALL_FACES), int(100*(i+1)/len(ALL_FACES))))
            sys.stdout.flush()

            x = self.create_pool_input(a, a_conv, b, n_faces, compute_method)
    
            threads = []
            for x_args in x:
                threads.append(self.single_approximation_parallel_process(x_args, barrier, queue, lock, r_queue))
            
            for t in threads:
                t.start()

            for t in threads:
                data = r_queue.get()
                if data[0] == "Approx":
                    exec_time_approx_sweep.append(data[1]*1000)
                if data[0] == "Compute":
                    exec_time_compute_sweep.append(data[1]*1000)
                if data[0] == "Sum":
                    exec_time_sum_sweep.append(data[1]*1000)
                    f_loss_results.append(data[2])
                    svd_comparison.append(self.compute.check_percent_svd(a, b, data[2]))
                    percent_approximated.append(int(100*n_faces/TOTAL_FACES))
                    results.append(data[3])
            
            for t in threads:
                t.join()
            

        #####################################################################################



        #####################################################################################
        print("")
        data = {
                    "exec-time-approx": exec_time_approx_sweep,
                    "exec-time-compute": exec_time_compute_sweep,
                    "exe-time-sum": exec_time_sum_sweep, 
                    "f-loss-results": f_loss_results,
                    "svd-comparison": svd_comparison,
                    "percent-approximated": percent_approximated,
                    "all-faces": ALL_FACES,
                    "a-shape": np.shape(a_orig),
                    "b-shape": np.shape(b_orig),
                    "compute-only": exec_time_compute_sweep[0],
                    "approx-only": exec_time_approx_sweep[-1],
                    "hybrid-best-exe": best_performance(exec_time_approx_sweep, exec_time_compute_sweep, percent_approximated)[0],
                    "hybrid-best-percent": best_performance(exec_time_approx_sweep, exec_time_compute_sweep, percent_approximated)[1],
                    "results": results
                }

        # self.v.visualize_sweep_approximation(data)
        # self.v.visualize_f_loss_vs_percent_svd(data)
        # self.v.visualize_ai(data)
    
        return data

    