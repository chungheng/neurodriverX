
import pycuda.gpuarray as garray

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

        self.num = [len(x) for x in output]
        self.offset = np.cumsum(num) - self.num[0]
        self.array = [x['array'] for x in output]
        self.index = [x['index'] for x in output]
        self.input = input

    def update(self):
        for i, (n, o) in enumerate(zip(self.num, self.offset)):
            if n == 0:
                continue
            total = 0.
            for j in range(o, o+n):
                total += self.array[j][self.index[j]]
            self.output[i] = total

class AggregatorGPU(AggregatorCPU):
    def __init__(self, input, output):
        """
        Aggregate outputs of units and feeds the result as input to other units.

        Parameters:
        output (list) : list of dict of the form {'index': ..., 'array': ...}.
        input (ndarray or pycuda.gpuarray) : input to a model.
        """

        super(AggregatorGPU, self).__init__(input=input, output=output)

        ptr = np.zeros(len(self.array), dtype=np.int64)

        for i, (arrary, index) in zip(self.array, self.index):
            ptr[i] = array.gpudata + array.dtype.itemsize*index

        self.ptr = garray.to_gpu(ptr)

        self.kernel = None

    def update(self):
        # TODO
        self.kernel.prepared_async_call()
