import sys
import copy
import time
import numpy as np
import multiprocessing

from lib.visualization import Visualization

class Experiments:
    def __init__(self, approx, comp):
        self.approx = approx
        self.comp = comp
        self.v = Visualization()
        self.a = None
        self.b = None

        self.manager = None
        self.thread_lock = None
        self.final_answer_lock = None
        self.a_shared = None
        self.b_shared = None
        self.b_rows_ordered = None

        # approx shared
        self.shared_lut = None
        self.shared_vals = None
        self.a_approx = None


    def single_approximation(self, a_orig, b_orig, n_faces=0):
        a = copy.deepcopy(a_orig)
        b = copy.deepcopy(b_orig)

        c_approx_zero, exe_time_approx = self.approx.cluster_inference(a[:, 0:n_faces].T)
        c_compute, exe_time_compute = self.comp.outer(a[:, n_faces:], b[n_faces:])
        c_final = c_compute + c_approx_zero

        f_loss = self.comp.check_accuracy(a, b, c_final, False)

        return exe_time_approx, exe_time_compute, f_loss

    def single_approximation_parallel(self, n_faces, n_threads):
        approx_threads = []
        compute_threads = []

        return_data = self.manager.dict({"face": list(np.zeros((np.shape(self.a_shared)[1], np.shape(self.b_shared)[1]))),
                                         "exe_time_approx": 0,
                                         "exe_time_compute": 0})

        id = 0
        # Create Approx Threads
        faces_per_thread = int(n_faces/n_threads)
        for t in range(n_threads):
            if (t+1) == n_threads:
                a_cols = np.arange(t*faces_per_thread, n_faces)
                b_rows = self.b_rows_ordered[t*faces_per_thread:n_faces]
            else:
                a_cols = np.arange(t*faces_per_thread, (t+1)*faces_per_thread)
                b_rows = self.b_rows_ordered[t*faces_per_thread:(t+1)*faces_per_thread]

            approx_threads.append(self.approx.cluster_inference_parallel(id, self.thread_lock, self.final_answer_lock,
                                                                         self.shared_lut, self.shared_vals, return_data,
                                                                         self.a_approx, self.b_shared,
                                                                         a_cols, b_rows))
            id += 1
        
        # Create Compute Threads
        faces_per_thread = int((np.shape(self.a_shared)[0] - n_faces)/n_threads)
        for t in range(n_threads):
            if (t+1) == n_threads:
                a_cols = np.arange(n_faces + t*faces_per_thread, np.shape(self.a_shared)[0])
                b_rows = self.b_rows_ordered[n_faces + t*faces_per_thread:]
            else:
                a_cols = np.arange(n_faces + t*faces_per_thread, n_faces + (t+1)*faces_per_thread)
                b_rows = self.b_rows_ordered[n_faces + t*faces_per_thread:n_faces + (t+1)*faces_per_thread]
            compute_threads.append(self.comp.outer_parallel(id, self.thread_lock, self.final_answer_lock, return_data,
                                                            self.a_shared, self.b_shared, a_cols, b_rows))
            id += 1

        ########################################################################
        # Run Approx Threads
        t1 = time.perf_counter()
        for t in approx_threads:
            t.start()

        for t in approx_threads:
            t.join()
        t2 = time.perf_counter()
        print("\nTime for Approx: %0.2fms" % ((t2 - t1) * 1000))
        ########################################################################
        
        ########################################################################
        # Run Compute Threads
        t1 = time.perf_counter()
        for t in compute_threads:
            t.start()

        for t in compute_threads:
            t.join()
        t2 = time.perf_counter()
        print("Time for Compute: %0.2fms" % ((t2 - t1) * 1000))
        ########################################################################

        c_final = return_data["face"]

        f_loss = self.comp.check_accuracy(self.a, self.b, c_final, False)
        #print("\nF-loss: %0.2f" % f_loss)
        exit()
        return return_data["exe_time_approx"], return_data["exe_time_compute"], f_loss


    def sweep_approximation(self, a_orig, b_orig, STEP=1, parallel=True, N_THREADS=2):
        TOTAL_FACES = np.shape(a_orig)[1]
        ALL_FACES = np.arange(0, TOTAL_FACES+1, STEP)

        if parallel:
            self.a = copy.deepcopy(a_orig)
            self.b = copy.deepcopy(b_orig)

            self.manager = multiprocessing.Manager()
            self.thread_lock = multiprocessing.Semaphore(value=N_THREADS)
            self.final_answer_lock = multiprocessing.Lock()
            self.shared_lut = self.manager.dict(self.approx.lut)
            self.shared_vals = self.manager.list(self.approx.vals)

            # possibly quantize a and b here
            self.a_shared = self.manager.list(copy.deepcopy(a_orig).T)
            self.a_approx = self.manager.list(self.approx.convert_data(copy.deepcopy(a_orig).T, len(a_orig)))
            self.b_shared = self.manager.list(copy.deepcopy(b_orig))

            # change this eventually for better accuracy!
            self.b_rows_ordered = np.arange(0, np.shape(self.b_shared)[1])

        # data to save
        f_loss_results = []
        exec_time_approx_sweep = []
        exec_time_compute_sweep = []
        percent_approximated = []

        for n_faces in ALL_FACES:
            sys.stdout.write(f'\rPercent Completed: {int(100*n_faces/TOTAL_FACES)}%')
            sys.stdout.flush()

            if parallel:
                data = self.single_approximation_parallel(n_faces=n_faces, n_threads=N_THREADS)
            else:
                data = self.single_approximation(a_orig, b_orig, n_faces=n_faces)

            exec_time_approx_sweep.append(data[0]*1000)
            exec_time_compute_sweep.append(data[1]*1000)
            f_loss_results.append(data[2]*1000)

            percent_approximated.append(int(100*n_faces/TOTAL_FACES))
        print("")
        data = {
                    "exec-time-approx": exec_time_approx_sweep,
                    "exec-time-compute": exec_time_compute_sweep,
                    "f-loss-results": f_loss_results,
                    "percent-approximated": percent_approximated,
                }

        self.v.visualize_sweep_approximation(data)
