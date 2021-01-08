#!/usr/bin/env python3
import argparse
import sys
import numpy
from pprint import pprint
from PIL import Image

from jpeg import JPEGEncoder, JPEGDecoder

assert sys.version_info >= (3, 6), 'Python 3.6+ is required'


def get_output_filename(filename, args):
    return f'{filename}-{args.m}-{args.scale}-{args.n}.pjpg'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['encode', 'decode', 'aio'], help='aio for all in one')
    parser.add_argument('--m', type=int)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('filename', type=str, nargs='+')
    args = parser.parse_args()
    pprint(args)

    if args.mode == 'encode':
        for filename in args.filename:
            print(f'Encoding {filename}')
            output_filename = get_output_filename(filename, args)
            JPEGEncoder(filename, args.m, args.scale, args.n).save(output_filename)
    elif args.mode == 'decode':
        for filename in args.filename:
            print(f'Decoding {filename}')

            with open(filename, 'rb') as f:
                JPEGDecoder(f).save(f'{filename}.jpg')
    elif args.mode == 'aio':
        for filename in args.filename:
            print(f'Encoding {filename}')
            output_filename = get_output_filename(filename, args)
            JPEGEncoder(filename, args.m, args.scale, args.n).save(output_filename)

            print(f'Decoding {output_filename}')

            with open(output_filename, 'rb') as f:
                JPEGDecoder(f).save(f'{output_filename}.jpg')

            signal = numpy.array(Image.open(filename))  ## input orignal data
            mean_signal = numpy.mean(signal)
            signal_diff = signal - mean_signal
            var_signal = numpy.sum(numpy.mean(signal_diff**2))  ## variance of orignal data

            noisy_signal = numpy.array(Image.open(f'{output_filename}.jpg')) ## input noisy data
            noise = noisy_signal - signal
            mean_noise = numpy.mean(noise)
            noise_diff = noise - mean_noise
            var_noise = numpy.sum(numpy.mean(noise_diff**2))    ## variance of noise

            if var_noise == 0:
                snr = 100    ## clean image
            else:
                snr = (numpy.log10(var_signal/var_noise))*10    ## SNR of the data
            print('SNR:', snr)
            
    else:
        raise ValueError(f'Unsupported mode {args.mode}')


if __name__ == '__main__':
    main()
