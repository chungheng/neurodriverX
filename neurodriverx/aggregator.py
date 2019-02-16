
import pycuda.gpuarray as garray

class Arr(object):
    def __init__(self, arr, idx):
        self.arr = arr
        self.idx = idx
    def __call__(self):
        return self.arr[self.idx]
    def __repr__(self):
        return repr(self.__call__())

class AggregatorCPU(object):
    def __init__(self, array, offset, num, output):
        """
        Parameters:
        arrays (list) : list of ndarray or pycuda.gpuarray
        indices (list) : index for the array
        offsets (list) :
        """
        self.array = array
        self.offset = offset
        self.num = num
        self.output = output

    def update(self):
        for i, (n, o) in enumerate(zip(self.num, self.offset)):
            if n == 0:
                continue
            total = 0.
            for j in range(o, o+n):
                total += self.array[j]()
            self.output[i] = total

class AggregatorGPU(object):
    def __init__(self, array, offset, num, output):
        """
        Parameters:
        arrays (list) : list of ndarray or pycuda.gpuarray
        indices (list) : index for the array
        offsets (list) :
        """
        self.array = array
        self.offset = offset
        self.num = num
        self.output = output

        ptr = np.zeros(len(self.array), dtype=np.int64)

        for i, arr in self.array:
            ptr[i] = arr.arr.gpudata + arr.arr.dtype.itemsize*arr.idx

        self.ptr = garray.to_gpu(ptr)

        self.kernel = None

    def update(self):
        # TODO
        self.kernel.prepared_async_call()
