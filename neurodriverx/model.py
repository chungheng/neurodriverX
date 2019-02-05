
import copy
import inspect
import os
import collections
from types import MethodType

from six import with_metaclass, get_function_globals, get_function_code, \
    StringIO

import numpy as np

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature

class _Variable(object):
    default = {
        'id': None,
        'type': None,
        'name': None,
        'value': None,
    }
    def __init__(self, **kwargs):
        for key, val in self.default.items():
            val = kwargs.pop(key, val)
            self.__dict__[key] = val
        if len(kwargs):
            raise AttributeError(', '.join(kwargs.keys()))
    def __repr__(self):
        return "{{type:{}, value:{}}}".format(self.type, self.value)
    def __setattribute__(self, key, val):
        if key not in self.__dict__:
            raise KeyError("Unrecognized key: {}".format(key))
        self.__dict__[key] = val

class VariableAnalyzer(CodeGenerator):
    """
    Analyze the variables in a set of ODEs
    """

    def __init__(self, func, defaults, **kwargs):
        self.func = func
        self.code = get_function_code(func)
        self.defaults = defaults

        self.globals = get_function_globals(self.func)

        inputs = self._extract_signature(func)
        self.variables = kwargs.pop('variables', dict())
        for key, val in inputs:
            self.variables[key] = _Variable(type='input', value=val, name=key)

        with open(os.devnull, 'w') as f:
            CodeGenerator.__init__(self, func, ostream=f)

            self.generate()

    def _extract_signature(self, func):
        """
        Extract the signature of a function
        """
        arg_spec = inspect.getfullargspec(func)
        args, defaults = arg_spec.args, arg_spec.defaults or list()

        arg_1st = args.pop(0)
        assert arg_1st == 'self', \
            "The first argument of the ode function must be 'self'"

        assert len(args) == len(defaults), \
            "Missing defaults for {}".format(args[0])
        return list(zip(args, defaults))

    def handle_load_fast(self, ins):
        """
        """
        key = ins.argval

        assert key == 'self' or key in self.variables or key in self.globals, \
            "Unrecognized variable {}".format(key)

        self.var.append( ins.argval )

    def handle_store_fast(self, ins):
        """
        """
        key = ins.argval
        if key not in self.variables:
            self._set_variable(key, type='local')
        self.var[-1] = "{} = {}".format(key, self.var[-1])

    def handle_load_attr(self, ins):
        """
        """
        key = ins.argval
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self._set_variable(key, type='state')
            elif key not in self.variables:
                self._set_variable(key, type='param')
            self.var[-1] = key
        else:
            self.var[-1] += "." + key

    def handle_store_attr(self, ins):
        """
        """
        key = ins.argval
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self._set_variable(key, type='state')
            elif key not in self.variables or \
                self.variables[key].type != 'state':
                self._set_variable(key, type='inter')
            self.var[-1] = key
        else:
            self.var[-1] += "{}.{}".format(self.var[-1], key)
        self.var[-2] = "{} = {}".format(self.var[-1], self.var[-2])
        del self.var[-1]

    def _set_variable(self, name, **kwargs):
        t = kwargs['type']
        if t != 'local':
            assert name in self.defaults, \
                "Missing default value for Variable {}".format(name)
        if name not in self.variables:
            val = self.defaults.get(name, None)
            self.variables[name] = _Variable(type=t, value=val, name=name)

        for key, val in kwargs.items():
            setattr(self.variables[name], key, val)

class FuncGenerator(CodeGenerator):
    def __init__(self, func, variables = None, backend=None, **kwargs):
        self.func = func
        self.code = get_function_code(func)
        self.backend = backend or None
        self.variables = variables or dict()

        CodeGenerator.__init__(self, self.func,  offset=4, ostream=StringIO())

    def generate(self):
        signature = ','.join(get_func_signature(self.func))
        fname = self.func.__name__
        self.ostream.write("def {}({}):\n".format(fname, signature))

        super(FuncGenerator, self).generate()

        return self.ostream.getvalue()

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == 'self':
            if 'd_' in key:
                key = key.split('d_')[-1]
                self.var[-1] += ".grad['%s']" % key
                return

            if key in self.variables:
                attr = self.variables[key].type
                self.var[-1] += ".%s['%s']" % (attr, key)
                return
        self.var[-1] += ".%s" % key

    def handle_store_attr(self, ins):
        self.handle_load_attr(ins)
        if self.backend is 'numpy':
            template = "{0}[:] = {1}"
        else:
            template = "{0} = {1}"
        self.var[-2] = template.format(self.var[-1], self.var[-2])
        del self.var[-1]


class CudaMetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2cu = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, '_source_funcs'):
                for func in val._source_funcs:
                    py2cu[func] = val
        dct['pyfunc_to_cufunc'] = py2cu
        return super(CudaMetaClass, cls).__new__(cls, clsname, bases, dct)


class CudaGenerator(with_metaclass(CudaMetaClass, CodeGenerator)):
    def __init__(self, model, **kwargs):
        pass

    def _post_output(self):
        self.newline = ';\n'
        CodeGenerator._post_output(self)

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == 'self':
            if key in self.model.Default_States:
                self.var[-1] = "states.{0}".format(key)
            elif key[:2] == 'd_' and key[2:] in self.model.Default_States:
                self.var[-1] = "gstates.{0}".format(key[2:])
            elif key in self.model.Default_Params:
                self.var[-1] = key.upper()
            elif key in self.model.Default_Inters:
                self.var[-1] = "inters.{0}".format(key)
        else:
            self.var[-1] = "{0}.{1}".format(self.var[-1], key)

    def process_jump(self, ins):
        if len(self.jump_targets) and self.jump_targets[0] == ins.offset:
            if len(self.var):
                self.output_statement()
            self.jump_targets.pop()
            self.space -= self.indent
            self.leave_indent = False

            self.var.append('}')
            self.newline = '\n'
            self.output_statement()

    def handle_store_fast(self, ins):
        if ins.argval == self.var[-1]:
            del self.var[-1]
            return
        if ins.argval not in self.variables and ins.argval not in self.signature:
            self.variables.append(ins.argval)
        self.var[-1] = "{0} = {1}".format(ins.argval, self.var[-1])

    def handle_binary_power(self, ins):
        self.var[-2] = "powf({0}, {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_and(self, ins):
        self.var[-2] = "({0} && {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_or(self, ins):
        self.var[-2] = "({0} || {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_call_function(self, ins):
        narg = int(ins.arg)

        # hacky way to handle keyword arguments
        if self.kwargs and self.var[-(narg+1)] == (self.kwargs + ".pop"):
            self.var[-(narg+1)] = self.var[-narg]
            self.signature.append(str(self.var[-narg]))
        else:
            args = [] if narg == 0 else list(map(str, self.var[-narg:]))
            func_name = self.var[-(narg+1)]
            pyfunc = eval(func_name, self.func_globals)
            cufunc = self.pyfunc_to_cufunc.get(pyfunc)
            if cufunc is not None:
                self.var[-(narg+1)] = cufunc(self, args)
            else:
                temp = ', '.join(args)
                self.var[-(narg+1)] = "{0}({1})".format(func_name, temp)

        if narg:
            del self.var[-narg:]

    def handle_pop_jump_if_true(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if (!{0}) {{".format(self.var[-1])
        self.newline = '\n'

    def handle_pop_jump_if_false(self, ins):
        self.jump_targets.append(ins.arg)
        self.enter_indent = True
        self.var[-1] = "if ({0}) {{".format(self.var[-1])
        self.newline = '\n'

    def handle_jump_forward(self, ins):
        self.leave_indent = True
        self.output_statement()

        target, old_target = ins.argval, self.jump_targets.pop()

        if target != old_target:
            self.newline = '\n'
            self.var.append("} else {")
            self.enter_indent = True
            self.jump_targets.append(target)
        else:
            self.var.append('}')
            self.newline = '\n'
            self.output_statement()

    def handle_return_value(self, ins):
        val = self.var[-1]
        if val is None:
            val = '0'
        self.var[-1] = "return {0}".format(val)


def _analyze_variable(func, defaults, variables):
    var_analyzer = VariableAnalyzer(func, defaults, variables=variables)
    return var_analyzer.variables

def _compile_func(func, variables, backend):

    codegen = FuncGenerator(func, variables=variables, backend=backend)
    src = codegen.generate()
    co = compile(src, '<string>', 'exec')
    locs = dict()
    globals = dict.copy(get_function_globals(func))
    eval(co, globals, locs)
    ode = locs[func.__name__]
    del locs

    return ode, src

class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):

        defaults = dct['defaults']
        bound = dict()

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
        for key in func_list:
            vars.update(_analyze_variable(dct[key], defaults, vars))

        # dct.update(vars)
        # dct['vars'] = {k:v for k, v in vars.items() if v.type != 'local'}
        dct['vars'] = vars
        dct.update({k: v for k, v in vars.items() if v.type != 'local'})

        for attr in ['inter', 'param', 'state', 'input']:
            dct[attr] = {k:v.value for k, v in vars.items() if v.type == attr}

        for key in func_list:
            func, src = _compile_func(dct[key], vars, dct['backend'])
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
        if var is None or var.type == 'local':
            raise AttributeError(key)
        dct = getattr(self, var.type)
        if hasattr(value, "__len__"):
            value = np.asarray(value)
        dct[key] = value

    def set_attrs(self, skip=False, **kwargs):
        for k, v in kwargs.items():
            if skip and (k not in self.vars or self.vars[k].type == 'local'):
                continue
            self[k] = v

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

        func_list = [x for x in ['ode', 'post'] if hasattr(self, x)]

        for f in func_list:
            func, src = _compile_func(getattr(self, f), self.vars, self.backend)
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

    def ode(self):
        pass

    def post(self):
        pass

    def update(self, dt, **kwargs):
        self._ode(**kwargs)

        for key, val in self.grad.items():
            self.state[key] += dt*val

        self._post()

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
        if key in model.vars and model.vars[key].type != 'local':
            var = copy.deepcopy(getattr(model, key))
            var.id = self.store['id']
            return var
        return super(modeldict, self).__getattribute__(key)
