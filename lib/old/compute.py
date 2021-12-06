import time
import multiprocessing
import numpy as np

class Compute:
    def __init__(self):
        pass

    def outer(self, A, B):
        C = np.zeros((np.shape(A)[0], np.shape(B)[1]))
        exe_time = 0
        for col_num in range(np.shape(A)[1]):
            t1 = time.perf_counter()
            C_new = np.outer(A[:, col_num], B[col_num])
            exe_time += time.perf_counter() - t1
            C += C_new
        return C, exe_time / 8

    def outer_parallel(self, id, thread_lock, final_answer_lock, return_data, a_shared, b_shared, a_cols, b_rows):
        return OuterThread(id, thread_lock, final_answer_lock, return_data, a_shared, b_shared, a_cols, b_rows)

    def check_accuracy(self, A, B, C_approx, print_ans=True):
        C_real = np.dot(A, B)
        f_norm = np.linalg.norm(C_real - C_approx, 'fro')
        if print_ans:
            print("Frobenius Norm: %0.2f" % f_norm)
        return f_norm

class OuterThread(multiprocessing.Process):
    def __init__(self, id, thread_lock, final_answer_lock, return_data, a_shared, b_shared, a_cols, b_rows):
        multiprocessing.Process.__init__(self)
        self.id = id
        self.thread_lock = thread_lock
        self.final_answer_lock = final_answer_lock
        self.return_data = return_data
        self.a_shared = a_shared
        self.b_shared = b_shared
        self.a_cols = a_cols
        self.b_rows = b_rows

    def run(self):
        exe_time = 0
        final = np.zeros((np.shape(self.a_shared)[1], np.shape(self.b_shared)[1]))
        #self.thread_lock.acquire()
        for compute_face in range(len(self.b_rows)):
            a_vec = np.array(self.a_shared[self.a_cols[compute_face]])
            b_vec = np.array(self.b_shared[self.b_rows[compute_face]])

            t1 = time.perf_counter()
            final += np.outer(a_vec, b_vec)
            t2 = time.perf_counter()
            exe_time += t2 - t1
        #self.thread_lock.release()

        '''
        self.final_answer_lock.acquire()
        #self.return_data["face"] += final
        if self.return_data["exe_time_compute"] < exe_time:
            self.return_data["exe_time_compute"] = exe_time
        self.final_answer_lock.release()
        '''