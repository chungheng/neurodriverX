
import inspect
import os

from six import with_metaclass, get_function_globals, get_function_code, \
    StringIO

from pycodegen.codegen import CodeGenerator
from pycodegen.utils import get_func_signature


def analyze_variable(func, defaults, variables):
    var_analyzer = VariableAnalyzer(func, defaults, variables=variables)
    return var_analyzer.variables, var_analyzer.locals

def compile_func(func, variables, backend):

    codegen = FuncGenerator(func, variables=variables, backend=backend)
    src = codegen.generate()
    co = compile(src, '<string>', 'exec')
    locs = dict()
    globals = dict.copy(get_function_globals(func))
    eval(co, globals, locs)
    ode = locs[func.__name__]
    del locs

    return ode, src

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
        self.locals = kwargs.pop('locals', dict())

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

        if key != 'self' and key not in self.variables and \
            key not in self.globals and key not in self.locals:
            "Variable '{}' is not defined".format(key)

        self.var.append( ins.argval )

    def handle_store_fast(self, ins):
        """
        """
        key = ins.argval
        if key not in self.variables:
            self.locals[key] = _Variable(name=key, type='local')
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
