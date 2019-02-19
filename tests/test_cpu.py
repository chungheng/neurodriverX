import unittest

import numpy as np

from neurodriverx.aggregator import AggregatorCPU

class GPUTestCase(unittest.TestCase):
    def test_aggregator_cpu(self):

        # create 10 output arrays
        output_num = 10
        output_num_len = np.random.randint(10, 30, output_num)
        output = [np.random.rand(x) for x in output_num_len]
        total_num = sum(output_num_len)

        # create the input array
        input_len = 50
        input = np.zeros(input_len)
        truth = np.zeros(input_len)

        # create output to input mapping
        io_mapping = []
        for i in range(input_len):
            num = np.random.randint(total_num)
            lst = []
            for j in range(num):
                p = np.random.randint(output_num)
                q = np.random.randint(output_num_len[p])
                lst.append({'array': output[p], 'index':q})
                truth[i] += output[p][q]
            io_mapping.append(lst)

        # test aggregator
        aggregator = AggregatorCPU(input=input, output=io_mapping)
        aggregator.update()

        assert np.allclose(truth, input)
