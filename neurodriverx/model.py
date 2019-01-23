import inspect
import os

from six import with_metaclass, get_function_globals, get_function_code

from pycodegen.codegen import CodeGenerator

class _Variable(object):
    default = {
        'type': None,
        'val': None,
    }
    def __init__(self, **kwargs):
        for key, val in self.default.items():
            val = kwargs.pop(key, val)
            self.__dict__[key] = val
        if len(kwargs):
            raise AttributeError(', '.join(kwargs.keys()))
    def __repr__(self):
        return "{{type:{}, value:{}}}".format(self.type, self.val)
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
        print(self.code)

        self.globals = get_function_globals(self.func)

        input = self._extract_signature(func)
        self.variables = {k:_Variable(type='input', val=v) for k,v in inputs}

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

    def handle_call_function(self, ins):
        narg = int(ins.arg)

        args = [] if narg == 0 else self.var[-narg:]
        func_name = self.var[-(narg+1)]
        self.var[-(narg+1)] = "{}({})".format(func_name , ','.join(args))

        if narg:
            del self.var[-narg:]

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
            self.variables[key] = _Variable(type='local')
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
                self._set_variable(key, type='parameter')
            self.var[-1] = key
        else:
            self.var[-1] += "." + ins.arg_name

    def handle_store_attr(self, ins):
        """
        """
        key = ins.argval
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key.split('d_')[-1]
                self._set_variable(key, type='state')
            elif key not in self.variables:
                self._set_variable(key, type='intermediate')
            self.var[-1] = key
        else:
            self.var[-1] += "{}.{}".format(self.var[-1], key)
        self.var[-2] = "{} = {}".format(self.var[-1], self.var[-2])
        del self.var[-1]

    def _set_variable(self, var_name, **kwargs):
        var_type = kwargs['type']
        if var_type != 'local':
            assert var_name in self.defaults, \
                "Missing default value for Variable {}".format(var_name)
        if var_name not in self.variables:
            self.variables[var_name] = _Variable(type=var_type)

        for key, val in kwargs.items():
            setattr(self.variables[var_name], key, val)


class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):

        ode = dct['ode']
        defaults = dct['defaults']
        # extract input argument of ode
        variables = cls._analyze_variable(func=ode, defaults=defaults)
        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)

    def _analyze_variable(func, defaults):
        var_analyzer = VariableAnalyzer(func, defaults)
        return var_analyzer.variables

class Model(with_metaclass(ModelMetaClass, object)):
    defaults = dict()
    def __init__(self):
        pass

    def ode(self):
        pass
