import unittest

import numpy as np

import pycuda
import pycuda.autoinit
import pycuda.gpuarray as garray

from neurodriverx.aggregator import AggregatorGPU

class GPUTestCase(unittest.TestCase):
    def test_aggregator_gpu(self):
        dtype = np.float64

        # create 10 output arrays
        output_num = 10
        output_num_len = np.random.randint(1, 3, output_num)
        output = [np.random.rand(x) for x in output_num_len]

        output_gpu = [garray.to_gpu(x) for x in output]
        total_num = sum(output_num_len)

        # create the input array
        input_len = 100
        input = np.zeros(input_len)
        input_gpu = garray.zeros(input_len, dtype=dtype)
        truth = np.zeros(input_len)

        # create output to input mapping
        io_mapping = []
        for i in range(input_len):
            num = np.random.randint(total_num)
            lst = []
            for j in range(num):
                p = np.random.randint(output_num)
                q = np.random.randint(output_num_len[p])
                lst.append({'array': output_gpu[p], 'index':q})
                truth[i] += output[p][q]
            io_mapping.append(lst)

        # test aggregator
        aggregator = AggregatorGPU(input=input_gpu, output=io_mapping)
        aggregator.update()

        assert np.allclose(truth, input_gpu.get())
