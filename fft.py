
import tensorflow as tf
import numpy as np
from threading import Thread

class FFT(Thread):

    def __init__(self, input_queue, output_queue, fft_params):
        Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.params = fft_params
        self.inbuf = [0.0]
        self.__keepThread = True

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_input = tf.placeholder(tf.float32)
            self.fft_mag = self._tf_fft_process(self.tf_input)
            self.graph.finalize()

    def _tf_fft_process(self, tf_input):
        with tf.device('/gpu:0'):
            stft = tf.contrib.signal.stft(tf_input, frame_length=self.params.fft_length, frame_step=self.params.fft_step, pad_end=True) # [channels, frames, ffts]
            stft = tf.reduce_mean(stft, axis=1) # [channels, ffts]

            mag = tf.abs(stft)
            mag = tf.reduce_mean(tf.contrib.signal.frame(mag, self.params.fft_average, 1, pad_end=True, pad_value=tf.reduce_mean(mag)), axis=-1) # Spatial fft average
            # mag = tf.subtract(mag[0], mag[1])

        phase = tf.angle(stft)
        phase = tf.reduce_mean(tf.contrib.signal.frame(phase, self.params.phase_smooth, 1, pad_end=True, pad_value=tf.reduce_mean(phase)), axis=-1)  # Spatial angle average
        # phase = tf.add(phase[0], phase[1])

        return mag, phase

    def run(self):
        with tf.Session(graph=self.graph) as sess:
            while self.__keepThread:
                pipe_dict = self.input_queue.get()
                f, p = sess.run(self.fft_mag, feed_dict={self.tf_input: pipe_dict['samples']})
                pipe_dict['fft/freq'] = f
                pipe_dict['fft/phase'] = p
                self.output_queue.put(pipe_dict)
