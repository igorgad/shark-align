
import queue
import jack_client
import fft
import display

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_channels', default=2, type=int, help='number of channels to use in the measure')
parser.add_argument('--fft_average', default=8, type=int, help='fft frequency average over octaves')
parser.add_argument('--fft_windowing', default=[], type=str, help='NOT IMPLEMENTED (hamming)')
parser.add_argument('--fft_accumulate', default=4096, type=int, help='accumulator before fft. Number of decay frames is given by accumulator samples / fft_length')
parser.add_argument('--fft_length', default=16384, type=int, help='fft frame size')
parser.add_argument('--fft_step', default=1024, type=int, help='fft step')

args = parser.parse_args(''.split())

jack_to_accumulator = queue.Queue(maxsize=4096)
accumulator_to_fft = queue.Queue(maxsize=4096)
fft_to_graphics = queue.Queue(maxsize=4096)

jcli = jack_client.JackClient(jack_to_accumulator, args.num_channels, 'shark')
acc = jack_client.Accumulator(jack_to_accumulator, accumulator_to_fft, args.fft_accumulate)
fftop = fft.FFT(accumulator_to_fft, fft_to_graphics, args.num_channels, fft_params=args)
render = display.Display(fft_to_graphics, jcli.client.samplerate, fft_params=args)

render.start()
fftop.start()
acc.start()
jcli.activate()
