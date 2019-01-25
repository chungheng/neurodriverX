import copy
import types
import networkx as nx
from collections import OrderedDict
from .model import Model

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
        self.Model_Defaults = {}

    def _get_model(self, model):

        if type(model) is str:
            models = get_all_subclasses(Model)
            if model in models:
                return models[model]
            else:
                raise TypeError("Unsupported model type {}".format(model))

        if isinstance(model, type):
            assert issubclass(model, Model)
        else:
            assert isinstance(model, Model)
            model = model.__class__

        return model

    def _parse_model_kwargs(self, obj, **kwargs):
        """
        Parameters
        ----------
        obj : str, instance of `Model`, or subclass of `Model`
            A node can be any hashable Python object except None.
        kwargs : dict
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.
        """
        model = self._get_model(obj)

        if model in self.Model_Defaults:
            model = self.Model_Defaults[model]

        obj = obj if isinstance(obj, Model) else model
        dct = {key:obj[key] for key in model.vars}
        dct['model'] = model

        for key, val in kwargs.items():
            assert key in dct
            dct[key] = val
        return dct

    def set_model_default(self, obj, **kwargs):
        model = self._get_model(obj)

        if isinstance(obj, Model):
            obj = copy.deepcopy(obj)
        else:
            obj = model()

        for key, val in kwargs.items():
            obj[key] = val

        self.Model_Defaults[model] = obj

    def add_neuron(self, node, model, **kwargs):
        """Add a single neuron.

        Parameters
        ----------
        node : node
            A node can be any hashable Python object except None.
        model : string, submodule of Model, or instance of Model
            Name or the Python class of a neuron model.
        kwargs : dict
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron(1, 'LeakyIAF')
        >>> G.add_neuron(2, HodgkinHuxley)
        >>> G.add_neuron(3, 'LeakyIAF', threshould=5)

        >>> hh = HodgkinHuxley(gNa = 32.)
        >>> G.add_neuron(3, hh)

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        """
        dct = self._parse_model_kwargs(model, **kwargs)

        self.graph.add_node(node, **dct)


    def add_synapse(self, node, model, source, target, **kwargs):
        """Add a single synapse.

        Parameters
        ----------
        node : hashable
            A node can be any hashable Python object except None.
        model : string or submodule of NDComponent
            Name or the Python class of a neuron model.
        source : hashable or None
            A source can be any hashable Python object except None. The hash
            value of the pre-synaptic neuron. If None, the edge between 'source'
            and 'node' will be omitted.
        target : hashable or None
            A target can be any hashable Python object except None. The hash
            value of the post-synaptic neuron. If None, the edge between
            'target' and 'node' will be omitted.
        params : dict
            Parameters of the neuron model.
        states : dict
            Initial values of the state variables of the neuron model.
        delay : float
            Delay between pre-synaptic neuron and synapse.
        kwargs:
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_neuron('2', 'HodgkinHuxley', states={'n':0., 'm':0., 'h':1.})
        >>> G.add_synapse('1->2', '1', '2', 'AlphaSynapse')

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        """
        dct = self._parse_model_kwargs(model, **kwargs)

        self.graph.add_node(node, **attrs)

        if source:
            self.graph.add_edge(source, node, **delay)
        if target:
            self.graph.add_edge(node, target)

    def _get_delay(self, attrs):
        delay = attrs.pop('delay', None)
        delay = {'delay': delay} if delay else dict({})
        return delay

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
