
import numpy as np
from threading import Thread


class Accumulator(Thread):
    __keepThread = True
    buffer = None

    def __init__(self, input_queue, output_queue, num_samples, blocking=True):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_samples = num_samples
        self.input_block_size = 0
        self.isblocking = blocking

    def run(self):
        pipe_dict = self.input_queue.get()
        self.input_block_size = pipe_dict['samples'].shape[1]
        self.buffer = np.zeros_like(pipe_dict['samples'])

        while self.__keepThread:
            if self.isblocking:
                self.buffer = np.delete(self.buffer, np.arange(self.buffer.shape[1]), axis=1)
            else:
                self.buffer = np.delete(self.buffer, np.arange(self.input_block_size), axis=1)

            while self.buffer.shape[1] < self.num_samples:
                pipe_dict = self.input_queue.get()
                self.buffer = np.concatenate([self.buffer, pipe_dict['samples']], axis=1)

            out_pipe_dict = {'samples': np.array(self.buffer)}
            self.output_queue.put(out_pipe_dict)
