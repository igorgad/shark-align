
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

        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 5))
        self.line_fft, = ax1.plot(logfreq, self._decibel(np.random.rand(frq.size)))
        self.line_phase, = ax2.plot(logfreq, np.random.randint(-5, 5, frq.size))
        ax1.set_title('FFT')
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(["%.2f" % x for x in np.exp(xticks)])

        self.input_queue = input_queue

    def _decibel(self, lin):
        return 20 * np.log10(lin / np.max(lin))

    def update_graphics(self, data):
        f, p = self.input_queue.get()

        self.line_fft.set_ydata(self._decibel(f[0][1:]))
        self.line_phase.set_ydata(p[0][1:])

        return self.line_fft, self.line_phase

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_graphics, blit=True, interval=0.0001, repeat=False, init_func=lambda: [self.line_fft, self.line_phase])
        plt.show()

    def stop(self):
        self.ani._stop()
