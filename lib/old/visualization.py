import numpy as np
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        self.colors = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"]
        self.colors = ["#A41034", "#4B4E6D", "#6A8D92", "#80B192", "#A1E887"]
    def visualize_sweep_approximation(self, data):
        exec_time_approx_sweep      = data["exec-time-approx"]
        exec_time_compute_sweep     = data["exec-time-compute"]
        exe_time_summing            = data["exe_time_summing"]
        f_loss_results              = data["f-loss-results"]
        percent_approximated        = data["percent-approximated"]

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

        plt.xticks(np.arange(len(pa))*ignore_interval, pa)
        plt.xlabel(r"Percent of Matrix A which is Approximated")
        plt.ylabel(r"Execution Time (ms)")
        plt.plot(x_pos, exec_time_approx_sweep, "o-", label="approximate", c=self.colors[0])
        plt.plot(x_pos, exec_time_compute_sweep, "o-", label="compute", c=self.colors[1])
        plt.plot(x_pos, exe_time_summing, "o-", label="summing", c=self.colors[2])
        plt.plot(x_pos, [exec_time_compute_sweep[0]] * len(x_pos), "--", label="baseline", c="black")
        plt.legend()
        plt.show()

