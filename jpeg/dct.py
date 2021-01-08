import math
import numpy

def dct_matrix(n):
    ret = numpy.empty((n, n))

    for k in range(n):
        for i in range(n):
            ret[k, i] = math.pi / n * (i + .5) * k

    ret = numpy.cos(ret)  
    ret[0] /= math.sqrt(2)  # X_0 /= sqrt(2)
    return ret * math.sqrt(2 / n)

def idct_matrix(n):
    ret = numpy.empty((n, n))

    for k in range(n):
        for i in range(n):
            ret[k, i] = math.pi / n * i * (k + .5)

    ret = numpy.cos(ret)
    ret[:, 0] /= math.sqrt(2)  # x_0 /= sqrt(2)
    return ret * math.sqrt(2 / n)

def dft_matrix(n):
    (x, y) = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
    omega = numpy.exp(-2 * numpy.pi * 1j / n)
    ret = numpy.power(omega, x * y)
    return ret

def idft_matrix(n):
    (x, y) = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
    omega = numpy.exp(2 * numpy.pi * 1j / n)
    ret = numpy.power(omega, x * y)
    return ret

DCT_COEFFICIENT_MATRIX = dct_matrix(8)
IDCT_COEFFICIENT_MATRIX = idct_matrix(8)

DFT_COEFFICIENT_MATRIX = dft_matrix(8)
IDFT_COEFFICIENT_MATRIX = idft_matrix(8)

def dct(arr):
    return (arr.T @ DCT_COEFFICIENT_MATRIX).T @ DCT_COEFFICIENT_MATRIX

def idct(arr):
    return (arr.T @ IDCT_COEFFICIENT_MATRIX).T @ IDCT_COEFFICIENT_MATRIX

def dft(arr):
    return (arr.T @ DFT_COEFFICIENT_MATRIX).T @ DFT_COEFFICIENT_MATRIX

def idft(arr):
    return (arr.T @ IDFT_COEFFICIENT_MATRIX).T @ IDFT_COEFFICIENT_MATRIX
