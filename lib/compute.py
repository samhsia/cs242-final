import os
import time
import copy
import torch
import multiprocessing
import numpy as np

class Compute:
    def __init__(self, dtype):
        self.dtype = dtype

    def outer(self, A, B):
        C = np.zeros((np.shape(A)[0], np.shape(B)[1]), dtype=self.dtype)
        exe_time = 0
        for col_num in range(np.shape(A)[1]):
            t1 = time.perf_counter()
            C_new = np.outer(A[:, col_num], B[col_num])
            exe_time += time.perf_counter() - t1
            C += C_new
        return C, exe_time / 8

    def check_accuracy(self, A, B, C_approx, print_ans=True):
        C_real = np.dot(A, B)
        f_norm = np.linalg.norm(C_real - C_approx, 'fro')
        if print_ans:
            print("Frobenius Norm: %0.2f" % f_norm)
        return f_norm
    
    def check_percent_svd(self, a, b, f_loss_target):
        u, s, vh = np.linalg.svd(b.astype(np.float))

        svd_loss = 0
        sl = -1
        while svd_loss <= f_loss_target and sl + np.shape(b)[0] > 0:
            c_svd = np.dot(a, np.dot(u[:, :sl] * s[:sl], vh[:sl,]))
            svd_loss = self.check_accuracy(a, b, c_svd, False)
            sl -= 1
        return 100*(np.shape(b)[0]+sl) / np.shape(b)[0]

    def outer_parallel(self, x, barrier, queue, lock):
        if len(list(os.sched_getaffinity(os.getpid()))) > 1:
            os.sched_setaffinity(0, { list(os.sched_getaffinity(os.getpid()))[1] })

        # parse input
        a = x[1]
        b = x[2]

        # initialize variables
        exe_time = 0
        face = np.zeros((np.shape(a)[0], np.shape(b)[1]), dtype=self.dtype)

        barrier.wait()
        for face in range(np.shape(a)[1]):
            a_vec = a[:, face]
            b_vec = b[face]

            t1 = time.perf_counter()
            face = np.outer(a_vec, b_vec)
            t2 = time.perf_counter()
            
            queue.put(["Compute", face])
            exe_time += t2-t1

        queue.put(["DONE", None])
        r_queue.put(["Compute", exe_time, None])
        return

    def blas_parallel(self, x, barrier, queue, lock, r_queue):
        if len(list(os.sched_getaffinity(os.getpid()))) > 1:
            os.sched_setaffinity(0, { list(os.sched_getaffinity(os.getpid()))[1] })

        # parse input
        a = x[1]
        b = x[2]

        barrier.wait()

        t1 = time.perf_counter()
        face = np.dot(a, b)
        t2 = time.perf_counter()
         
        queue.put(["Compute", face])
        exe_time = (t2-t1)

        queue.put(["DONE", None])
        
        r_queue.put(["Compute", exe_time, None])
        return
    
    def blas_gpu_parallel(self, x, barrier, queue, lock, r_queue):
        if len(list(os.sched_getaffinity(os.getpid()))) > 1:
            os.sched_setaffinity(0, { list(os.sched_getaffinity(os.getpid()))[1] })

        # parse input
        a = torch.tensor(x[1])
        b = torch.tensor(x[2])
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        barrier.wait()
        
        start.record()
        face = torch.matmul(a.cuda(), b.cuda())
        face = face.to(device="cpu")
        end.record()
        
        torch.cuda.synchronize()
        
        queue.put(["Compute", np.array(face)])
        exe_time = start.elapsed_time(end) / 1000

        queue.put(["DONE", None])
        
        r_queue.put(["Compute", exe_time, None])
        return

    def sum_parallel(self, x, barrier, queue, lock, r_queue):
        if len(list(os.sched_getaffinity(os.getpid()))) > 1:
            os.sched_setaffinity(0, { list(os.sched_getaffinity(os.getpid()))[2] })

        # parse input
        a = x[1]
        b = x[2]
        QUANTIZE_FACE = x[3]
        approx = x[4]

        # initialize variables
        exe_time = 0 
        done_messages = 0
        face = np.zeros((np.shape(a)[0], np.shape(b)[1]))
        face_approx = np.zeros((np.shape(a)[0], np.shape(b)[1]), dtype=int)

        barrier.wait()

        while True:
            if done_messages == 2:
                if QUANTIZE_FACE:
                    face_approx = face_approx.flatten() # creating the new numpy datastructure should not be timed
                    t1 = time.perf_counter()
                    face_approx = approx.dequantize_row(face_approx)
                    t2 = time.perf_counter()
                    
                    face_approx = face_approx.reshape(np.shape(face)) # reshape should not be timed

                    t3 = time.perf_counter()
                    face = np.add(face, face_approx)
                    t4 = time.perf_counter()

                    exe_time += (t2-t1) + (t4-t3)
                f_loss = self.check_accuracy(a, b, face, False)
                r_queue.put(["Sum", exe_time, f_loss])
                return
            
            msg = queue.get()
            if msg[0] == "DONE":
                done_messages += 1
                continue
            else:
                new_face = np.array(msg[1])
            
            if QUANTIZE_FACE:
                if msg[0] == "Approx":
                    t1 = time.perf_counter()
                    face_approx = np.add(face_approx, new_face)
                    t2 = time.perf_counter()
                if msg[0] == "Compute":
                    t1 = time.perf_counter()
                    face = np.add(face, new_face)
                    t2 = time.perf_counter()
            else:
                t1 = time.perf_counter()
                face = np.add(face, new_face)
                t2 = time.perf_counter()
            exe_time += t2-t1

