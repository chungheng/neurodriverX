
import copy
import inspect
import os
import collections
from types import MethodType

from six import with_metaclass, get_function_globals, get_function_code, \
    StringIO

from pycodegen.codegen import CodeGenerator

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
        self.variables = {}
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
            elif key not in self.variables:
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

class OdeGenerator(CodeGenerator):
    def __init__(self, func, variables, **kwargs):
        self.func = func
        self.code = get_function_code(func)
        self.variables = variables

        CodeGenerator.__init__(self, self.func,  offset=4, ostream=StringIO())

    def generate(self):
        signature = inspect.signature(self.func)
        self.ostream.write("def ode{}:\n".format(str(signature)))

        super(OdeGenerator, self).generate()

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

class ModelMetaClass(type):
    def __new__(cls, clsname, bases, dct):

        defaults = dct['defaults']
        # extract variables from member functions
        vars = {}
        for key in ['ode']:
            func = dct[key]
            vars.update(cls._analyze_variable(func, defaults))

        for key in defaults:
             assert key in vars, "Unused variable {} in {}".format(key, clsname)

        dct.update(vars)
        dct['vars'] = [v.name for v in vars.values() if v.type != 'local']

        for attr in ['inter', 'param', 'state', 'local', 'input']:
            dct[attr] = {k:v.value for k,v in vars.items() if v.type == attr}

        _ode, _ode_src = cls._generate_executabale_ode(func, vars)
        dct['_ode'] = _ode
        dct['_ode_src'] = _ode_src

        return super(ModelMetaClass, cls).__new__(cls, clsname, bases, dct)

    def _analyze_variable(func, defaults):
        var_analyzer = VariableAnalyzer(func, defaults)
        return var_analyzer.variables

    def _generate_executabale_ode(func, variables):

        codegen = OdeGenerator(func, variables=variables)
        src = codegen.generate()
        co = compile(src, '<string>', 'exec')
        locs = dict()
        globals = dict.copy(get_function_globals(func))
        eval(co, globals, locs)
        ode = locs['ode']
        del locs

        return ode, src

    def __getitem__(cls, key):
        var = getattr(cls, key, None)
        if var is None:
            raise AttributeError(key)
        dct = getattr(cls, var.type)
        return dct[key]

class Model(with_metaclass(ModelMetaClass, object)):
    defaults = dict()

    def __init__(self,  **kwargs):
        self.id = kwargs.pop('id', '')

        self.param = self.__class__.param.copy()
        self.inter = self.__class__.inter.copy()
        self.state = self.__class__.state.copy()
        self.input = self.__class__.input.copy()
        self.grad = {key:0. for key in self.__class__.state.keys()}

        for key, val in kwargs.items():
            self[key] = val

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

    def ode(self):
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
        model = self.store['model']
        if key in model.vars:
            var = copy.deepcopy(getattr(model, key))
            var.id = self.store['id']
            return var
        return super(modeldict, self).__getattribute__(key)
