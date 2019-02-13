import inspect

from six import StringIO, get_function_globals, get_function_code
from six import with_metaclass
from functools import wraps
import random
import numpy as np
from jinja2 import Template

from pycodegen.codegen import CodeGenerator

template_cuda_kernel = Template("""
{{ preprocessing_definition }}
{{ struct_definition|join("\n") }}
{{ clip_definition }}
{{ device_function_definition|join("\n") }}
{{ solver_defintion }}

__global__ void {{ model_name }} (
    {{ kernel_arguments|join(",\n    ") }}
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    {{ struct_declaration|join("\n    ") }}

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        {{ import_data|join("\n        ") }}

        {{ sovler_invocation }}

        {{ post_invocation }}

        {{ export_data|join("\n        ") }}
    }

    return;
}
"""
)

template_define_struct = Template("""
struct {{ name }} {
    {%- for key in dct %}
    {{ dtype }} {{ key }};
    {%- endfor %}
};
""")

template_define_preprocessing = Template("""
/* Define constant parameters */
{% for key, val in param_constant.items() -%}
#define  {{ key.upper() }}\t\t{{ val }}
{% endfor -%}
/* Define upper and lower bounds of state variables */
{%- for key, val in bound.items() %}
#define  {{ key.upper() }}_MIN\t\t{{ val[0] }}
#define  {{ key.upper() }}_MAX\t\t{{ val[1] }}
{%- endfor -%}

""")

template_clip = Template("""
__device__ void clip(States &state)
{
    {%- for key, val in bound.items() %}
    state.{{ key }} = fmax{{ float_char }}(state.{{ key }}, {{ key.upper() }}_MIN);
    state.{{ key }} = fmin{{ float_char }}(state.{{ key }}, {{ key.upper() }}_MAX);
    {%- endfor %}
}
""")

template_forward_euler = Template("""
__device__ int forward(
    {{ dtype }} dt,
    {{ signature|join(",\n    ") }}
)
{
    {{ ode_invocation }}
    {% for key in state %}
    state.{{ key }} += dt * grad.{{ key }};
    {%- endfor %}
    {{ clip_invocation }}
}
""")

template_device_func = Template("""
__device__ int {{ name }}(
    {{ signature|join(",\n    ") }}
)
{
    {%- for var in local %}
    {{ dtype }} {{ var }};
    {%- endfor %}

{{ src -}}
}
""")

class MetaClass(type):
    def __new__(cls, clsname, bases, dct):
        py2cu = dict()
        for key, val in dct.items():
            if callable(val) and hasattr(val, '_source_funcs'):
                for func in val._source_funcs:
                    py2cu[func] = val
        dct['pyfunc_to_cufunc'] = py2cu
        return super(MetaClass, cls).__new__(cls, clsname, bases, dct)

class CudaFuncGenerator(with_metaclass(MetaClass, CodeGenerator)):
    """
    Generate CUDA device function, ex. ode() and post().
    """
    def __init__(self, func, local, variables, param, dtype=None, **kwargs):
        self.func = func
        self.code = get_function_code(func)
        self.dtype = 'float' if dtype == np.float32 else 'double'
        self.local = local
        self.param = param
        self.variables = variables
        self.float_char = 'f' if dtype == np.float32 else ''
        self.func_globals = get_function_globals(self.func)

        self.used_variables = set()

        CodeGenerator.__init__(self, self.func, newline=';\n', offset=4, \
            ostream=StringIO())

        self.generate()
        self.generate_signature()

        fname = self.func.__name__

        self.definition = template_device_func.render(
            dtype = self.dtype,
            name = fname,
            local = self.local,
            signature = self.signature,
            used = self.used_variables,
            src = self.ostream.getvalue()
        )
        self.invocation = "{}({});".format(fname, ", ".join(self.args))

    def _post_output(self):
        self.newline = ';\n'
        CodeGenerator._post_output(self)

    def generate_signature(self):

        self.signature = []
        self.args = []
        if "state" in self.used_variables:
            self.signature.append("State &state")
            self.args.append("state")
        if "grad" in self.used_variables:
            self.signature.append("State &grad")
            self.args.append("grad")
        if "inter" in self.used_variables:
            self.signature.append("Inter &inter")
            self.args.append("inter")

        for key, val in self.param.items():
            if key in self.used_variables and hasattr(val, '__len__'):
                msg = "const {} {}".format(self.dtype, key.upper())
                self.signature.append(msg)
                self.args.append(key.upper())

        full_args = inspect.getargspec(self.func)
        num_offset = len(full_args.args or []) - len(full_args.defaults or [])
        for arg in full_args.args[num_offset:]:
            self.signature.append("{} {}".format(self.dtype, arg))
            self.args.append(arg)

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

    def handle_load_attr(self, ins):
        key = ins.argval
        if self.var[-1] == 'self':
            if key[:2] == 'd_':
                key = key[2:]
                assert key in self.variables
                self.var[-1] = "grad.{0}".format(key)
                self.used_variables.add('grad')
            else:
                assert key in self.variables
                t = self.variables[key].type
                if t == 'param':
                    self.var[-1] = key.upper()
                    self.used_variables.add(key)
                else:
                    self.used_variables.add(t)
                    self.var[-1] = "{0}.{1}".format(t, key)
        else:
            self.var[-1] = "{0}.{1}".format(self.var[-1], key)

    def handle_store_fast(self, ins):
        if ins.argval == self.var[-1]:
            del self.var[-1]
            return
        self.var[-1] = "{0} = {1}".format(ins.argval, self.var[-1])

    def handle_binary_and(self, ins):
        self.var[-2] = "({0} && {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_or(self, ins):
        self.var[-2] = "({0} || {1})".format(self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_binary_power(self, ins):
        msg = "pow{0}({1}, {2})"
        self.var[-2] = msg.format(self.float_char, self.var[-2], self.var[-1])
        del self.var[-1]

    def handle_call_function(self, ins):
        narg = int(ins.arg)

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

    def _cuda_func_factory(self, func, args):
        return "{0}({1})".format(func, ', '.join(args))

    def _random_func(func):
        """
        A decorator for registering random functions
        """
        @wraps(func)
        def wrap(self, args):
            self.has_random = True
            return func(self, args)
        return wrap

    def _py2cuda(*source_funcs):
        def wrapper(func):
            func._source_funcs = source_funcs
            return func
        return wrapper

    @_py2cuda(np.abs)
    def _np_abs(self, args):
        return self._cuda_func_factory('abs', args)

    @_py2cuda(np.exp)
    def _np_exp(self, args):
        return self._cuda_func_factory('exp' + self.float_char, args)

    @_py2cuda(np.power)
    def _np_power(self, args):
        return self._cuda_func_factory('pow' + self.float_char, args)

    @_py2cuda(np.cbrt)
    def _np_cbrt(self, args):
        return self._cuda_func_factory('cbrt' + self.float_char, args)

    @_py2cuda(np.sqrt)
    def _np_sqrt(self, args):
        return self._cuda_func_factory('sqrt' + self.float_char, args)

    @_py2cuda(random.gauss, np.random.normal)
    @_random_func
    def _random_gauss(self, args):
        func = 'curand_normal(&seed)'

        return "({0}+{1}*{2})".format(args[0], args[1], func)

    @_py2cuda(random.uniform)
    @_random_func
    def _random_uniform(self, args):
        func = 'curand_uniform(&seed)'

        if len(args) == 1:
            func = "({0}*{1})".format(args[0], func)
        elif len(args) == 2:
            func = "({0}+({1}-{0})*{2})".format(args[0], args[1], func)

        return func

class CudaKernelGenerator(object):
    """
    Generate CUDA kernel, ex. ode() and post().
    """

    def __init__(self, instance, dtype=np.float64):
        self.instance = instance
        self.dtype = 'float' if dtype == np.float32 else 'double'

        self.funcs = {x: None for x in ['ode', 'post'] if hasattr(instance, x)}

        self.param_constant = {}
        self.param_nonconst = {}
        for key, val in self.instance.param.items():
            if hasattr(val, '__len__'):
                self.param_nonconst[key] = val
            else:
                self.param_constant[key] = val

        for name in funcs.keys():
            f = getattr(instance, name)
            funcs[name] = CudaFuncGenerator(f , instance.locals[name],
                instance.vars, self.param_nonconst)

        src_preprocessing = generate_preprocessing()
        src_struct_definition, src_struct_declaration = generate_struct()
        src_clip_definition, src_clip_invocation = generate_clip()
        src_forward_euler_definition, src_forward_euler_invocation = generate_forward(func['ode'], src_clip_invocation)
        src_import, src_export, src_signature = generate_signature_import_export()

        self.src_kernel_definition = template_cuda_kernel.render(
            preprocessing_definition = src_preprocessing,
            struct_definition = src_struct_definition,
            clip_definition = src_clip_definition,
            device_function_definition = [f.definition for f in funcs.values()],
            solver_defintion = src_forward_euler_definition,
            model_name = instance.__class__.__name__,
            kernel_arguments = src_signature,
            struct_declaration = src_struct_declaration,
            import_data = src_import,
            sovler_invocation = src_forward_euler_invocation,
            post_invocation = funcs['post'].invocation,
            export_data = src_export
        )

    def generate_preprocessing(self):
        src = template_define_preprocessing.render(
            param_constant = param_constant,
            bound = self.instance.bound
        )
        return FuncSrc(src, None)

    def generate_struct(self):
        """
        Generate structure definition.
        """
        src_struct_definition = []
        src_struct_declaration = []
        if self.instance.state:
            src = template_define_struct.render(
                name = 'State',
                dct = self.instance.state,
                dtype = self.dtype
            )
            src_struct_definition.append(src)
            src_struct_declaration.append("State state, grad;")

        if self.instance.inter:
            src = template_define_struct.render(
                name = 'Inter',
                dct = self.instance.inter,
                dtype = self.dtype
            )
            src_struct_definition.append(src)
            src_struct_declaration.append("Inter inter;")
        return src_struct_definition, src_struct_declaration

    def generate_clip(self):
        """
        Generate clip definition.
        """
        src_clip_definition = ""
        src_clip_invocation = ""
        if self.instance.bound:
            src_clip_definition = template_clip.render(
                bound = self.instance.bound
            )
            src_clip_invocation = "clip(state);"
        return src_clip_definition, src_clip_invocation

    def generate_forward(self, ode, src_clip_invocation):
        src_forward_euler_definition = template_forward_euler.render(
            dtype = self.dtype,
            signature = ode.signature,
            ode_invocation = ode.invocation,
            clip_invocation = src_clip_invocation,
            state = self.instance.state
        )
        src_forward_euler_invocation = "forward(dt, {});".format(
            ", ".join(ode.args))
        return src_forward_euler_definition, src_forward_euler_invocation

    def generate_signature_import_export(self):
        src_import, src_export = [], []
        src_signature = ["int num_tread", "{} dt".format(self.dtype)]

        for key in self.instance.state.keys():
            src_signature.append( "{} *g_{}".format(self.dtype, key) )
            src_export.append( "g_{0}[tid] = state.{0};".format(key) )
            src_import.append( "state.{0} = g_{0}[tid];".format(key) )

        for key in self.instance.inter.keys():
            src_signature.append( "{} *g_{}".format(self.dtype, key) )
            src_export.append( "g_{0}[tid] = inter.{0};".format(key) )
            src_import.append( "inter.{0} = g_{0}[tid];".format(key) )

        for key in param_nonconst.keys():
            src_signature.append( "{} *g_{}".format(self.dtype, key.upper()) )
            src = "{0} {1} = g_{1}[tid];".format(self.dtype, key.upper())
            src_import.append( src )

        for key in self.instance.input.keys():
            src_signature.append( "{} *g_{}".format(self.dtype, key.upper()) )
            src_import.append( "{0} {1} = g_{1}[tid];".format(self.dtype, key) )
        return src_import, src_export, src_signature
