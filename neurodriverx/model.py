
import copy
import inspect
import os
import collections
from types import MethodType

from six import with_metaclass, get_function_globals

import numpy as np

from .codegen import _Variable, VariableAnalyzer, FuncGenerator, \
    analyze_variable, compile_func

class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):

        defaults = dct['defaults']
        bound = dict()

        # extract bound from defaults
        for key, val in defaults.items():
            if hasattr(val, '__len__'):
                assert len(val) == 3, "Variable {} ".format(key) + \
                    "should be a scalar of a iterable of 3 elements " + \
                    "(initial value, upper bound, lower bound), " + \
                    "but {} is given.".format(val)
                defaults[key] = val[0]
                bound[key] = val[1:]

        dct['bound'] = bound

        if 'backend' not in dct:
            dct['backend'] = 'scalar'
        # extract variables from member functions
        func_list = [x for x in ['ode', 'post'] if x in dct]

        vars = {}
        locals = {}
        for key in func_list:
            new_vars, new_locals = analyze_variable(dct[key], defaults, vars)
            vars.update(new_vars)
            locals[key] = new_locals

        dct['locals'] = locals
        dct['vars'] = vars
        dct.update(vars)

        for attr in ['inter', 'param', 'state', 'input']:
            dct[attr] = {k:v.value for k, v in vars.items() if v.type == attr}

        for key in func_list:
            func, src = compile_func(dct[key], vars, dct['backend'])
            fname = '_{}_{}'.format(dct['backend'], key)
            dct[fname] = func
            dct[fname+'_src'] = src
            dct['_'+key] = dct[fname]

        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)

    def __getitem__(cls, key):
        var = getattr(cls, key, None)
        if var is None:
            raise AttributeError(key)
        dct = getattr(cls, var.type)
        return dct[key]

class Model(with_metaclass(ModelMetaClass, object)):
    defaults = dict()
    backend = 'scalar'

    def __init__(self, **kwargs):
        self.id = kwargs.pop('id', '')
        self.backend = kwargs.pop('backend', self.__class__.backend)

        self.param = self.__class__.param.copy()
        self.inter = self.__class__.inter.copy()
        self.state = self.__class__.state.copy()
        self.input = self.__class__.input.copy()
        self.bound = self.__class__.bound.copy()
        self.grad = {key:0. for key in self.__class__.state.keys()}

        bound = kwargs.pop('bound', dict())
        self.set_bounds(**bound)
        self.set_attrs(**kwargs)

    def __getitem__(self, key):
        var = getattr(self, key, None)
        if var is None:
            raise AttributeError(key)
        dct = getattr(self, var.type)
        return dct[key]

    def __setitem__(self, key, value):
        var = getattr(self, key, None)
        if var is None:
            raise AttributeError(key)
        dct = getattr(self, var.type)
        dct[key] = value

    def set_attrs(self, skip=False, **kwargs):
        for key, val in kwargs.items():
            if skip and key not in self.vars:
                continue
            self[key] = val

    def set_bounds(self, **kwargs):
        for key, val in kwargs.items():
            if val not in self.state:
                raise KeyError("Only state variable has bounds: {}".format(key))
            if not hasattr(val, __len__) or len(val) != 2:
                raise ValueError("Bounds of {} should follow ".format(key) + \
                    "the format (lower bound, upper bound), " + \
                    "but {} is provided.".format(key))
            self.bound[key] = val

    def compile(self, backend=None):
        self.backend = backend or self.backend

        if self.backend == 'pycuda':
            self._compile_gpu()
        else:
            self._compile_cpu()

    def _compile_cpu(self):
        func_list = [x for x in ['ode', 'post'] if hasattr(self, x)]

        for f in func_list:
            func, src = compile_func(getattr(self, f), self.vars, self.backend)
            fname = '_{}_{}'.format(self.backend, f)
            func = MethodType(func, self)
            setattr(self, fname, func)
            setattr(self, fname+'_src', src)
            setattr(self, '_'+f, func)

        for k in self.grad:
            if self.backend == 'scalar':
                self.grad[k] = 0.
            elif self.backend == 'numpy':
                self.grad[k] = np.zeros_like(self.state[k])

    def _compile_gpu(self):
        pass

    def ode(self):
        pass

    def post(self):
        pass

    def _update_cpu(self, dt, **kwargs):
        self._ode(**kwargs)

        for key, val in self.grad.items():
            self.state[key] += dt*val

        self._post()

    def _update_gpu(self, d_t):
        self.gpu.kernel.prepared_async_call(
            self.gpu.grid,
            self.gpu.block,
            None,
            d_t*self.time_scale,
            self.gpu.num,
            *self.gpu.arg_address)

    def update(self, dt, **kwargs):
        pass


class modeldict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        if 'id' not in self.store:
            self.store['id'] = None

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr(self.store)

    def __getattr__(self, key):
        if key == 'id':
            return self['id']
        model = self.store['model']
        if key in model.vars:
            var = copy.deepcopy(getattr(model, key))
            var.id = self.store['id']
            return var
        return super(modeldict, self).__getattribute__(key)
