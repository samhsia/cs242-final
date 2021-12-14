import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        self.colors = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]
        self.colors = ["#A41034", "#4B4E6D", "#6A8D92", "#80B192", "#A1E887"]
        
    def visualize_sweep_approximation(self, data):
        exec_time_approx_sweep      = data["exec-time-approx"]
        exec_time_compute_sweep     = data["exec-time-compute"]
        exe_time_sum                = data["exe-time-sum"]
        percent_approximated        = data["percent-approximated"]
        a_shape                     = data["a-shape"]
        b_shape                     = data["b-shape"]

        x_pos = np.arange(len(percent_approximated))

        if len(percent_approximated) > 20:
            pa = []
            ignore_interval = int(len(percent_approximated) / 20) + 1
            for dot_val in percent_approximated:
                if dot_val % ignore_interval == 0:
                    pa.append(dot_val)
        else:
            ignore_interval = 1
            pa = percent_approximated

        plt.xticks(np.arange(len(pa))*ignore_interval, pa, fontsize=14)
        plt.xlabel(r"Percent of Matrix A which is Approximated", fontsize=14)
        plt.ylabel(r"Execution Time (ms)", fontsize=14)
        plt.plot(x_pos, exec_time_approx_sweep, "o-", label="approximate", c=self.colors[0])
        plt.plot(x_pos, exec_time_compute_sweep, "o-", label="compute", c=self.colors[1])
        plt.plot(x_pos, exe_time_sum, "o-", label="summing", c=self.colors[2])
        plt.plot(x_pos, [exec_time_compute_sweep[0]] * len(x_pos), "--", label="baseline", c="black")
        plt.title("batch size (" + str(a_shape[0]) + ") dimension (" + str(b_shape[0]) + ")", fontsize=14)
        plt.legend()
        plt.show()


    def visualize_f_loss_vs_percent_svd(self, data):
        svd_comparison              = data["svd-comparison"]
        f_loss_results              = data["f-loss-results"]
        percent_approximated        = data["percent-approximated"]
        a_shape                     = data["a-shape"]
        b_shape                     = data["b-shape"]                   

        fig, ax1 = plt.subplots()

        x_pos = np.arange(len(percent_approximated))
        color = self.colors[0]
        ax1.set_xticks(np.arange(len(percent_approximated)))
        ax1.set_xticklabels(percent_approximated)
        ax1.set_xlabel(r"Matrix Approximation, %", fontsize=12)
        ax1.set_ylabel(r"Approximation Error, F-Loss", fontsize=12, color=color)
        ax1.plot(x_pos, f_loss_results, "o-", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(r'Aproximation Error for $M=256$ and $N=64$', fontsize=12)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = self.colors[1]
        ax2.set_ylabel('percent svd', fontsize=12, color=color)  # we already handled the x-label with ax1
        ax2.plot(x_pos, svd_comparison, "o-", color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("F-loss and Percent SVD vs. Percent Approximated")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        #plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig("./f-loss-percent-svd.png", dpi=700)
        plt.show()

    def visualize_ai(self, data):
        percent_approximated        = data["percent-approximated"]
        all_faces                   = data["all-faces"]
        a_shape                     = data["a-shape"]
        b_shape                     = data["b-shape"]
        
        NUM_FACES = a_shape[1]
        ai_blas = []
        ai_approx = []
        ai_net = []
        for approx_faces in all_faces:
            compute_faces = (NUM_FACES - approx_faces)

            # Compute AI 
            compute_blas    = compute_faces * (2 * (a_shape[0] * b_shape[1]))                       # multiplying vectors and adding faces together
            io_blas         = compute_faces * (1 * (a_shape[0] + b_shape[1]))                       # reading vectors from memory

            # Approx AI
            compute_approx  = approx_faces * (a_shape[0] * b_shape[1])                              # summing the faces
            io_approx       = approx_faces * (a_shape[0] * b_shape[1]) + approx_faces * a_shape[0]  # reading faces from memory, reading col values of a

            ai_blas.append(compute_blas/io_blas if io_blas != 0 else None)
            ai_approx.append(compute_approx/io_approx if io_approx != 0 else None)
            ai_net.append((compute_blas + compute_approx) / (io_blas + io_approx))
        


        x_pos = np.arange(len(percent_approximated))
        plt.xlabel(r"Percent of Matrix A which is Approximated", fontsize=12)
        plt.xticks(np.arange(len(percent_approximated)), percent_approximated)
        plt.ylabel(r"Arithmetic Intensity", fontsize=12)

        plt.plot(x_pos, ai_blas, c=self.colors[0], label="BLAS AI")
        plt.plot(x_pos, ai_approx, c=self.colors[1], label="Approx AI")
        plt.plot(x_pos, ai_net, c=self.colors[2], label="Net AI")
        plt.legend()
        plt.show()