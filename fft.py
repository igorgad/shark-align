
import tensorflow as tf
import numpy as np
from threading import Thread

class FFT(Thread):

    def __init__(self, input_queue, output_queue, num_channels, fft_params):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.params = fft_params
        self.num_channels = num_channels
        self.inbuf = [0.0]
        self.__keepThread = True
        self.input_fn = tf.py_func(self.__dequeue, [], tf.float32)
        self.sess = tf.Session()

    def __dequeue(self):
        self.inbuf = self.input_queue.get()
        return np.array(self.inbuf, np.float32) # [channels, samples]

    def __enqueue(self, val):
        return self.output_queue.put(val)

    def _getinputs(self):
        return self.sess.run(self.input_fn)

    def _tf_fft_process(self):
        tf_input = self.input_fn # [channels, samples]

        with tf.device('/gpu:0'):
            stft = tf.contrib.signal.stft(tf_input, frame_length=self.params.fft_length, frame_step=self.params.fft_step, pad_end=True) # [channels, frames, ffts]
            stft = tf.reduce_mean(stft, axis=1) # [channels, ffts]

            mag = tf.abs(stft)
            mag = tf.reduce_mean(tf.contrib.signal.frame(mag, self.params.fft_average, 1, pad_end=True), axis=-1) # Spatial fft average
        # angle = tf.reduce_mean(tf.contrib.signal.frame(tf.angle(stft), self.params.fft_average, 1, pad_end=True), axis=-1)  # Spatial angle average
        return mag, tf.angle(stft)

    def run(self):
        g = tf.Graph()
        with g.as_default():
            fft_mag = self._tf_fft_process()
            g.finalize()

        while self.__keepThread:
            f, p = self.sess.run(fft_mag)
            self.__enqueue([f, p])
