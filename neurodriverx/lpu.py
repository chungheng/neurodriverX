import copy
from types import SimpleNamespace
from collections import OrderedDict
from .model import Model, modeldict, _Variable
import numbers

import networkx as nx
import numpy as np
import pycuda
import pycuda.gpuarray as garray

from .aggregator import Arr, AggregatorCPU, AggregatorGPU

def get_all_subclasses(cls):
    all_subclasses = {}

    for subclass in cls.__subclasses__():
        all_subclasses[subclass.__name__] = subclass
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses

class LPU(object):
    """ Local Processing Unit

    LPU is an abstract of a neural circuit consisting of neurons and synapses.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph.node_attr_dict_factory = modeldict
        self.Model_Defaults = {}

    def _get_class(self, obj):
        """
        Parameters
        ----------
        obj : str, instance of `Model`, or subclass of `Model`
        """
        if type(obj) is str:
            classes = get_all_subclasses(Model)
            if obj in classes:
                return classes[obj]
            else:
                raise TypeError("Unsupported model type {}".format(obj))

        if isinstance(obj, type):
            assert issubclass(obj, Model)
            return obj
        else:
            assert isinstance(obj, Model)
            return obj.__class__

    def _get_model_instance(self, obj, **kwargs):
        """
        Parameters
        ----------
        obj : str, instance of `Model`, or subclass of `Model`
            A node can be any hashable Python object except None.
        kwargs : dict
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Returns
        -------
        new_obj : an instnace of Model
        """
        id = kwargs.pop('id', '')
        _class = self._get_class(obj)

        if isinstance(obj, Model):
            new_obj = copy.deepcopy(obj)
        elif _class in self.Model_Defaults:
            new_obj = copy.deepcopy(self.Model_Defaults[_class])
        else:
            new_obj = _class()

        new_obj.set_attrs(**kwargs)

        return new_obj

    def _get_model_attributes(self, obj, **kwargs):
        """
        Parameters
        ----------
        obj : str, instance of `Model`, or subclass of `Model`
            A node can be any hashable Python object except None.
        kwargs : dict
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Returns
        -------
        new_obj : an instnace of Model
        """
        _class = self._get_class(obj)

        if isinstance(obj, Model):
            new_obj = obj
        elif _class in self.Model_Defaults:
            new_obj = self.Model_Defaults[_class]
        else:
            new_obj = _class

        attr = {'model': _class, 'id': kwargs.pop('id', '')}
        for key in new_obj.vars:
            val = kwargs.pop(key, new_obj[key])
            if not isinstance(val, numbers.Number):
                raise ValueError("Variable {} should be a number".format(key))
            attr[key] = val

        if len(kwargs):
            raise KeyError(kwargs.keys())

        return attr

    def set_model_default(self, obj, **kwargs):
        attr = self._get_model_attributes(obj, **kwargs)

        self.Model_Defaults[attr['model']] = attr

    def add(self, id, obj, inputs=None, outputs=None, **kwargs):
        """Add a single synapse.

        Parameters
        ----------
        id : str
        obj : string, subclass of Model, or instance of Model
            Name or the Python class of a neuron model.
        inputs :
        outputs :
        kwargs:
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Examples
        --------
        >>> g = LPU()
        >>> g.add('1', 'HodgkinHuxley')
        >>> g.add('2', HodgkinHuxley, v=3)
        >>> hh = HodgkinHuxley(v=4)
        >>> neu = g.add('3', hh)

        >>> g.add('1-2', AMPA, inputs='1.spike', outputs='2.I')
        >>> g.add('3-2', AMPA, inputs=neu.spike, outputs='2.I')
        >>> g.add('3-1', AMPA, inputs=neu.spike, outputs='1')
        >>> g.add('2-3', AMPA, inputs='2', outputs=neu)

        Notes
        -----
        """

        attr = self._get_model_attributes(obj, id=id, **kwargs)

        self.graph.add_node(id, **attr)

        if inputs:
            self.set_inputs(id, inputs)

        if outputs:
            self.set_outputs(id, outputs)

        return self.graph.nodes[id]

    def _parse_id_and_attribute(self, obj, attr=None):
        if type(obj) == str:
            if obj in self.graph.nodes:
                id = obj
            else:
                seg = obj.split('.')
                id = '.'.join(seg[:-1])
                attr = seg[-1]
        elif isinstance(obj, modeldict):
            id = obj.id
        elif isinstance(obj, _Variable):
            id = obj.id
            attr = obj.name
        else:
            raise TypeError("Unsupported type of {}".format(obj))

        if not id in self.graph.nodes:
            raise KeyError("Graph has no {}".format(id))

        return id, attr

    def _validate_model_attribute(self, model, candidates, attr):
        if attr not in candidates:
            msg = "'{}' does not appear in '{}'".format(attr, model.__name__)
            raise AttributeError(msg)

    def _set_single_input(self, iid, oid, iattr=None, oattr=None):

        iid, iattr = self._parse_id_and_attribute(iid, iattr)
        oid, oattr = self._parse_id_and_attribute(oid, oattr)
        imodel = self.graph.nodes[iid]['model']
        omodel = self.graph.nodes[oid]['model']

        icandidates = list(imodel.input.keys())
        ocandidates = list(omodel.inter.keys()) + list(omodel.state.keys())

        if iattr is None and oattr is None:
            attr = [x for x in icandidates if x in ocandidates]

            if len(attr) == 0:
                raise AttributeError("Fail to infer I/O pair;" + \
                    " {} and {}".format(imodel.__name__, omodel.__name__) + \
                    " have no common attribute.")

            elif len(attr) > 1:
                raise AttributeError("Fail to infer I/O pair;" + \
                    " {} and {}".format(imodel.__name__, omodel.__name__) + \
                    " have more than one common attributes" + \
                    " {}".format(attr))

            iattr = oattr = attr[0]

        elif iattr is None and oattr is not None:
            self._validate_model_attribute(imodel, icandidates, oattr)
            iattr = oattr

        elif iattr is not None and oattr is None:
            self._validate_model_attribute(omodel, ocandidates, iattr)
            oattr = iattr

        self._validate_model_attribute(imodel, icandidates, iattr)
        self._validate_model_attribute(omodel, ocandidates, oattr)

        self.graph.add_edge(oid, iid, output=oattr, input=iattr)

    def set_inputs(self, id, inputs):

        if type(inputs) is list or type(inputs) is tuple:
            for val in inputs:
                self._set_single_input(id, val)
        elif type(inputs) is dict:
            for key, val in inputs.items():
                self._set_single_input(id, val, iattr=key)
        else:
            self._set_single_input(id, inputs)

    def set_outputs(self, id, outputs):

        if type(outputs) is list or type(outputs) is tuple:
            for val in outputs:
                self._set_single_input(val, id)
        elif type(outputs) is dict:
            for key, val in outputs.items():
                self._set_single_input(val, id, oattr=key)
        else:
            self._set_single_input(outputs, id)

    def _serialize_model_attributes(self):
        self.models = {}

        for _id, data in self.graph.nodes(data=True):
            model = data['model']
            if model not in self.models:
                dct = {key:[] for key in data.keys() if key != 'model'}
                self.models[model] = dct

            for key, val in data.items():
                if key != 'model':
                    self.models[model][key].append(val)

        # reduce list to scalar if all entires are equal
        for model, dct in self.models.items():
            for key, lst in dct.items():
                if key in model.vars and \
                    model.vars[key].type == 'param' and \
                    lst.count(lst[0]) == len(lst):
                    dct[key] = lst[0]

    def _serialize_model_inputs(self):
        for dct in self.models.values():
            dct['id2idx'] = {x:i for i,x in enumerate(dct['id'])}
            dct['input'] = dict()

        for u, v, data in self.graph.edges(data=True):
            omodel = self.graph.nodes[u]['model']
            imodel = self.graph.nodes[v]['model']
            odct, idct = self.models[omodel], self.models[imodel]

            arr = {'array':odct[data['output']], 'index':odct['id2idx'][u]}

            iattr = data['input']
            if iattr not in idct['input']:
                idct['input'][iattr] = [list() for _ in idct['id']]
            idx = idct['id2idx'][v]
            idct['input'][iattr][idx].append(arr)

    def _allocate_model_memory(self):
        for model, dct in self.models.items():
            for key in model.vars:
                val = dct[key]
                if hasattr(val, '__len__'):
                    arr = np.asarray(val, dtype=self.dtype)
                    if self.backend == 'numpy':
                        dct[key] = arr
                    elif self.backend == 'pycuda':
                        dct[key] = garray.to_gpu(arr)

    def _instantiate_model(self):
        for model, dct in self.models.items():
            instance = model()
            instance.set_attrs(**self.models[model], skip=True)
            instance.compile(backend=self.backend)
            dct['instance'] = instance

    def _instantiate_aggregator(self):
        Agg = AggregatorGPU if self.backend == 'pycuda' else AggregatorCPU

        for model, dct in self.models.items():
            dct['aggregator'] = {}
            for key, val in dct['input'].items():
                dct['aggregator'][key] = Agg(input=dct[key], output=val)

    def compile(self, backend='numpy', dtype=np.float64):
        self.backend = backend
        self.dtype = dtype

        self._serialize_model_attributes()

        self._allocate_model_memory()

        self._instantiate_model()

        self._serialize_model_inputs()

        self._instantiate_aggregator()

    def _aggregate_input(self):
        for dct in self.models.values():
            for val in dct['aggregator'].items():
                val.update()

    def update(self, dt):
        self._aggregate_input()

        for dct in self.models.values():
            instance = dct['instance']
            instance.update(dt, **instance.input)

    def write_gexf(self, filename):
        graph = nx.MultiDiGraph()
        for n,d in self.graph.nodes(data=True):
            data = d.copy()
            if data['class'] != u'Port':
                model = data.pop('class')
                data['class'] = model.__name__
            graph.add_node(n, **data)
        for u,v,d in self.graph.edges(data=True):
            data = d.copy()
            graph.add_edge(u, v, **data)
        nx.write_gexf(graph, filename)

    def read_gexf(self, filename):
        self.graph = nx.MultiDiGraph()
        graph = nx.read_gexf(filename)
        for n,d in graph.nodes(data=True):
            if d['class'].encode('utf-8') == u'Port':
                self.add_port(n, **d)
            else:
                model = d.pop('class')
                # neuron and synapse are ambigious at this point
                self.add_neuron(n, model, **d)
        for u,v,d in graph.edges(data=True):
            self.graph.add_edge(u, v, **d)

    @property
    def neuron(self):
        n = {x:d for x,d in self.graph.nodes(True) if self.isneuron(d)}
        return n

    def neurons(self, data=False):
        n = [(x,d) for x,d in self.graph.nodes(True) if self.isneuron(d)]
        if not data:
            n = [x for x,d in n]
        return n

    def isneuron(self, n):
        return issubclass(n['class'], BaseAxonHillockModel.BaseAxonHillockModel) or \
            issubclass(n['class'], BaseMembraneModel.BaseMembraneModel)

    @property
    def synapse(self):
        # TODO: provide pre-/post- neuron hash value
        n = {x:d for x,d in self.graph.nodes(True) if self.issynapse(d)}
        return n

    def synapses(self, data=False):
        # TODO: provide pre-/post- neuron hash value
        n = [(x,d) for x,d in self.graph.nodes(True) if self.issynapse(d)]
        if not data:
            n = [x for x,d in n]
        return n

    def issynapse(self, n):
        return issubclass(n['class'], BaseSynapsekModel.BaseSynapsekModel)


if __name__ == "__main__":

    a = LPU()
    a.add_neuron('1', 'LeakyIAF')
    a.add_neuron('2', 'LeakyIAF')
    a.add_neuron('3', LeakyIAF)
    a.add_port('1',port='so', selector='xx')
    a.add_synapse('1--2', '1', '2', 'AlphaSynapse')
    a.write_gexf('temp.gexf')

    b = LPU()
    b.read_gexf('temp.gexf')
