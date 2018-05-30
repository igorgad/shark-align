
import jack
import numpy as np
from threading import Thread


class JackClient():

    __keepThread = True

    def __init__(self, output_queue, num_channels, client_name):
        self.output_queue = output_queue
        self.name = client_name
        self.num_channels = num_channels
        self._create_client()

    def _process(self, frames):
        self.output_queue.put_nowait(np.array([np.array(port.get_array()[:], np.float32) for port in self.client.inports], np.float32)) # [channels, samples]

    def _create_client(self):
        self.client = jack.Client(self.name)
        [self.client.inports.register('input_' + str(i)) for i in range(self.num_channels)]

        self.client.set_process_callback(self._process)

    def activate(self):
        self.client.activate()

    def deactivate(self):
        self.client.deactivate()

    def close(self):
        self.client.close()


class Accumulator(Thread):
    __keepThread = True

    def __init__(self, input_queue, output_queue, num_samples):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_samples = num_samples

    def run(self):
        while self.__keepThread:
            buf = self.input_queue.get()
            while buf.shape[1] < self.num_samples:
                buf = np.concatenate([buf, self.input_queue.get()], axis=1)

            self.output_queue.put(buf)
