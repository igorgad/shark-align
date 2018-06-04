
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue

class Display():

    def __init__(self, input_queue, sample_rate, fft_params):
        n = fft_params.fft_length
        k = np.arange(n)
        T = n / sample_rate
        frq = k / T  # two sides frequency range
        frq = frq[range(1, 1 + n // 2)]  # one side frequency range
        logfreq = np.log(frq)
        xticks = np.linspace(logfreq[0], logfreq[-1], 10)

        self.fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(14, 8))
        self.line_corr, = ax0.plot(np.arange(4096*2-1), np.random.rand(4096*2-1))
        self.line_fft, = ax1.plot(logfreq, self._decibel(np.random.rand(frq.size)))
        self.line_phase, = ax2.plot(logfreq, np.random.randint(-5, 5, frq.size))


        ax1.set_xticks(xticks)
        ax1.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])

        self.input_queue = input_queue

    def _decibel(self, lin):
        return 20 * np.log10(lin / np.max(lin))
        # return 20 * np.log10(lin / np.max(lin))

    def update_graphics(self, data):
        pipe_dict = self.input_queue.get()
        f = pipe_dict['fft/freq']
        p = pipe_dict['fft/phase']
        corr = pipe_dict['correlations']

        fy = f[0, 1:] - f[1, 1:]
        fy = fy + np.min(fy) + 1e-6
        corry = (corr[1] - np.min(corr[1])) / (np.max(corr[1] - np.min(corr[1])))

        self.line_corr.set_ydata(corry)
        self.line_fft.set_ydata(self._decibel(fy))
        self.line_phase.set_ydata(p[1,1:])

        return self.line_corr, self.line_fft, self.line_phase

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_graphics, blit=True, interval=0.0001, repeat=False, init_func=lambda: [self.line_fft, self.line_phase])
        plt.show()

    def stop(self):
        self.ani._stop()
