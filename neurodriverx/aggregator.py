
import numpy as np
from jinja2 import Template

import pycuda.driver as drv
import pycuda.gpuarray as garray

from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

template_gpu_aggregate = Template("""
__global__ void aggregate(
    int num_thread,
    {{ dtype }} **output,
    int *g_num,
    int *g_offset,
    {{ dtype }} *g_input)
{
    // must use block size (32, 32, 1)
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;
    int uid;

    __shared__ {{ dtype }} input[32][33];
    __shared__ int s_num[32];
    __shared__ int s_offset[32];
    int r_num, r_offset;
    {{ dtype }} *ptr;

    if (tidy == 0) {
        uid = bid * 32 + tidx;
        if (uid < num_thread) {
            s_num[tidx] = g_num[uid];
            s_offset[tidx] = g_offset[uid];
        }
    }
    input[tidy][tidx] = 0.0;
    __syncthreads();

    uid = bid * 32 + tidy ;
    if (uid < num_thread) {
        r_num = s_num[tidy];
        r_offset = s_offset[tidy];
        for (int i = r_offset+tidx; i < r_offset+r_num; i += 32) {
            ptr = output[i];
            input[tidy][tidx] += *ptr;
       }
    }

    __syncthreads();
    if (tidy < 8) {
        input[tidx][tidy] += input[tidx][tidy + 8];
        input[tidx][tidy] += input[tidx][tidy + 16];
        input[tidx][tidy] += input[tidx][tidy + 24];
    }
    __syncthreads();
    if (tidy < 4) {
        input[tidx][tidy] += input[tidx][tidy + 4];
    }
    __syncthreads();
    if (tidy < 2) {
        input[tidx][tidy] += input[tidx][tidy + 2];
    }
    __syncthreads();

    if (tidy == 0) {
        input[tidx][0] += input[tidx][1];
        uid = bid * 32 + tidx;
        r_num = s_num[tidx];
        if (uid < num_thread && r_num > 0)
            g_input[uid] = input[tidx][0];
    }
    return;
}
""")

class ProxyArray(object):
    def __init__(self, arr, idx):
        self.array = array
        self.idx = idx
    def __call__(self):
        return self.arr[self.idx]
    def __repr__(self):
        return repr(self.__call__())

class AggregatorCPU(object):
    def __init__(self, input, output):
        """
        Aggregate outputs of units and feeds the result as input to other units.

        Parameters:
        output (list) : list of dict of the form {'index': ..., 'array': ...}.
        input (ndarray or pycuda.gpuarray) : input to a model.
        """

        self.num = np.asarray([len(x) for x in output])
        self.offset = np.concatenate(([0], np.cumsum(self.num[:-1])))
        merged = sum(output, [])
        self.array = [x['array'] for x in merged]
        self.index = [x['index'] for x in merged]
        self.input = input

    def update(self):
        for i, (n, o) in enumerate(zip(self.num, self.offset)):
            if n == 0:
                continue
            total = 0.
            for j in range(o, o+n):
                total += self.array[j][self.index[j]]
            self.input[i] = total

class AggregatorGPU(AggregatorCPU):
    def __init__(self, input, output):
        """
        Aggregate outputs of units and feeds the result as input to other units.

        Parameters:
        output (list) : list of dict of the form {'index': ..., 'array': ...}.
        input (ndarray or pycuda.gpuarray) : input to a model.
        """

        super(AggregatorGPU, self).__init__(input=input, output=output)
        dtype = 'float' if self.input.dtype == np.float32 else 'double'

        ptr = np.zeros(len(self.array), dtype=np.int64)
        for i, (array, index) in enumerate(zip(self.array, self.index)):
            ptr[i] = int(array.gpudata) + array.dtype.itemsize*index

        self.ptr = garray.to_gpu(ptr)
        self.offset = garray.to_gpu(self.offset.astype(np.int32))
        self.num = garray.to_gpu(self.num.astype(np.int32))
        self.num_thread = len(self.input)


        src = template_gpu_aggregate.render(dtype = dtype)
        try:
            mod = SourceModule(src, options = ["--ptxas-options=-v"])
            kernel = mod.get_function('aggregate')
            kernel.prepare('iPPPP')
        except:
            for i, line in enumerate(src.split('\n')):
                print("{}: {}".format(i, line))
            raise

        self.kernel = kernel

        self.block = (32, 32, 1)
        self.grid = (int((self.num_thread - 1) / 32) + 1, 1)

    def update(self):
        # TODO
        self.kernel.prepared_async_call(
            self.grid,
            self.block,
            None,
            self.num_thread,
            self.ptr.gpudata,
            self.num.gpudata,
            self.offset.gpudata,
            self.input.gpudata
        )
