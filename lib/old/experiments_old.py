import os
import sys
import copy
import time
import numpy as np
import multiprocessing

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
        os.sched_setaffinity(0, {1, 3, 5}) # possibly need to set the 0 to my_pid

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


    def sweep_approximation_pool(self, a_orig, b_orig, STEP=1, compute_method="outer"):
        #####################################################################################
        # define constants
        TOTAL_FACES = np.shape(a_orig)[1]
        ALL_FACES = np.arange(0, TOTAL_FACES+1, STEP)

        a = a_orig
        a_conv = self.approx.convert_data(copy.deepcopy(a_orig))
        b = b_orig

        # create datastructures for saving stats
        exec_time_approx_sweep = []
        exec_time_compute_sweep = []
        exec_time_sum_sweep = []
        f_loss_results = []
        svd_comparison = []
        percent_approximated = []

        # shared datastructure for processing in parallel
        lock = multiprocessing.Lock()           # only used for testing... may remove later
        barrier = multiprocessing.Barrier(1)    # align start of compute process, approx process, sum process
        queue = multiprocessing.Queue()         # feed the sum process from compute or approx
        #####################################################################################


        #####################################################################################
        # sweep the number of faces using a Pool
        with multiprocessing.Pool(processes=3, initializer=self._pool_init, initargs=(lock, barrier, queue,)) as pool:
            for n_faces in ALL_FACES:
                sys.stdout.write(f'\rPercent Completed: {int(100*n_faces/TOTAL_FACES)}%')
                sys.stdout.flush()

                x = self.create_pool_input(a, a_conv, b, n_faces, compute_method)
                results = pool.map_async(self.single_approximation_parallel, x)
                data = results.get()
                
                exec_time_approx_sweep.append(data[0][0]*1000)
                exec_time_compute_sweep.append(data[1][0]*1000)
                exec_time_sum_sweep.append(data[2][0]*1000)
                f_loss_results.append(data[2][1])
                svd_comparison.append(self.compute.check_percent_svd(a, b, data[2][1]))
                percent_approximated.append(int(100*n_faces/TOTAL_FACES))
            pool.close()
            pool.join()

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
                }

        self.v.visualize_sweep_approximation(data)
        self.v.visualize_f_loss_vs_percent_svd(data)
        #self.v.visualize_ai(data)