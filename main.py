
import queue
import jack_client
import accumulator
import delay_tracker
import fft
import display

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_channels', default=2, type=int, help='number of channels to use in the measure')
parser.add_argument('--fft_average', default=8, type=int, help='fft frequency average over octaves')
parser.add_argument('--accumulator', default=4096, type=int, help='accumulator before fft. Number of decay frames is given by accumulator samples / fft_length')
parser.add_argument('--fft_length', default=16384, type=int, help='fft frame size')
parser.add_argument('--fft_step', default=256, type=int, help='fft step')
parser.add_argument('--phase_smooth', default=4, type=int, help='phase smoothing')

args = parser.parse_args(''.split())

jack_to_accumulator = queue.Queue(maxsize=128)
accumulator_to_delay_tracker = queue.Queue(maxsize=128)
delay_tracker_to_fft = queue.Queue(maxsize=128)
fft_to_graphics = queue.Queue(maxsize=128)

jcli = jack_client.JackClient(jack_to_accumulator, args.num_channels, 'shark')
acc = accumulator.Accumulator(jack_to_accumulator, accumulator_to_delay_tracker, args.accumulator)
dtracker = delay_tracker.DelayTracker(accumulator_to_delay_tracker, delay_tracker_to_fft)
fftop = fft.FFT(delay_tracker_to_fft, fft_to_graphics, fft_params=args)
render = display.Display(fft_to_graphics, jcli.client.samplerate, fft_params=args)

render.start()
fftop.start()
dtracker.start()
acc.start()
jcli.activate()
