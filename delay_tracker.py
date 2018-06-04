
import numpy as np
from threading import Thread


class DelayTracker(Thread):
    __keepThread = True
    __update = True

    def __init__(self, input_queue, output_queue):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.correlations = None
        self.delays = None
        self.nchannels = 0
        self.input_block_size = 0

    def update(self):
        self.reset_delay()
        __update = True

    def reset_delay(self):
        self.correlations = np.zeros([self.nchannels, 2 * self.input_block_size - 1])
        self.delays = np.zeros([self.nchannels, 1])

    def compute_delay(self, pipe_dict):
        corr = np.zeros([self.nchannels, 2*self.input_block_size-1])
        sdelay = np.zeros([self.nchannels, 1])
        for c in range(self.nchannels):
            corr[c] = np.correlate(pipe_dict['samples'][0], pipe_dict['samples'][c], "full")
            sdelay[c] = np.argmax(corr) - corr.size // 2

        self.correlations = (self.correlations + corr) / 2.0
        self.delays = (self.delays + sdelay) / 2.0

    def run(self):
        pipe_dict = self.input_queue.get()
        self.nchannels = pipe_dict['samples'].shape[0]
        self.input_block_size = pipe_dict['samples'].shape[1]

        self.reset_delay()

        while self.__keepThread:
            pipe_dict = self.input_queue.get()

            if self.__update:
                self.compute_delay(pipe_dict)

            pipe_dict['correlations'] = self.correlations
            pipe_dict['samples_delays'] = self.delays
            self.output_queue.put(pipe_dict)
