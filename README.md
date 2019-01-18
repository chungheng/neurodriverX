# NeurodriverX: a python package for simulating neural circuit on single GPU

The NeurodriverX is a Python implementation of a Local Processing Unit (LPU)  Local Processing Unit (LPU), based on the [Neurodriver](https://github.com/neurokernel/neurodriver) project under the umbrella of the [Neurokernel](https://github.com/neurokernel/neurokernel) open initiative. Unlike the original Neurodriver project which runs under the Neurokernel framework with support of MPI for multi-GPU simulation, NeurodriverX focuses on stand-alone single-GPU execution with capabilities of

  * High-level API for specification of neural circuit.
  * Automatic CUDA kernel generation for neural models defined in Python.
