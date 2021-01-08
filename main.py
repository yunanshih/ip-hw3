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
            JPEGEncoder(filename, args.m, args.scale, args.n).save(f'./output/{output_filename}')
    elif args.mode == 'decode':
        for filename in args.filename:
            print(f'Decoding {filename}')

            with open(filename, 'rb') as f:
                JPEGDecoder(f).save(f'./output/{filename}.jpg')
    elif args.mode == 'aio':
        for filename in args.filename:
            print(f'Encoding {filename}')

            encoder = JPEGEncoder(filename, args.m, args.scale, args.n)
            output_filename = get_output_filename(filename, args)
            encoder.save(f'./output/{output_filename}')

            print(f'Decoding {output_filename}')

            with open(f'./output/{output_filename}', 'rb') as f:
                JPEGDecoder(f).save(f'./output/{output_filename}.jpg')

            original = numpy.array(Image.open(filename))
            compressed = numpy.array(Image.open(f'./output/{output_filename}.jpg'))
            noise = compressed - original
            
            mean_original = numpy.mean(original)
            original_diff = original - mean_original
            var_original = numpy.sum(numpy.mean(original_diff**2))
           
            mean_noise = numpy.mean(noise)
            noise_diff = noise - mean_noise
            var_compressed = numpy.sum(numpy.mean(noise_diff**2))

            if var_compressed == 0:
                snr = 100
            else:
                snr = (numpy.log10(var_original/var_compressed))*10
            rmse = numpy.sqrt(((compressed - original) ** 2).mean())
            print('SNR:', snr)
            print('RMSE:', rmse)
            
    else:
        raise ValueError(f'Unsupported mode {args.mode}')

if __name__ == '__main__':
    main()
