
import jack
import numpy as np


class JackClient():

    __keepThread = True

    def __init__(self, output_queue, num_channels, client_name):
        self.output_queue = output_queue
        self.name = client_name
        self.num_channels = num_channels
        self._create_client()

    def _process(self, frames):
        pipe_dict = {'samples': np.array([np.array(port.get_array()[:], np.float32) for port in self.client.inports], np.float32)}  # [channels, samples]
        self.output_queue.put_nowait(pipe_dict)

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
