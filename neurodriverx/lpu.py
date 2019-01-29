import copy
import types
import networkx as nx
from collections import OrderedDict
from .model import Model, modeldict, _Variable
import numbers

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

        for key, val in kwargs.items():
            assert key in new_obj.vars
            new_obj[key] = val

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

        attr = {key: kwargs.pop(key, new_obj[key]) for key in new_obj.vars}
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

    def add_neuron(self, node, obj, **kwargs):
        """Add a single neuron.

        Parameters
        ----------
        name : str
            A node can be any hashable Python object except None.
        obj : string, submodule of Model, or instance of Model
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
        new_obj = self._get_model_instance(obj, **kwargs)

        for key, val in kwargs.items():
            assert key in new_obj.vars
            new_obj[key] = val

        self.graph.add_node(node, model=new_obj)

        return new_obj

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
        new_obj = self._get_model_instance(obj, **kwargs)

        for key, val in kwargs.items():
            if isinstance(val, numbers.Number):
                assert key in new_obj.vars
                new_obj[key] = val
            elif key in new_obj.input:
                self._set_input_to_node(val, new_obj, None, key)
            elif key in new_obj.state or key in new_obj.inter:
                self._set_input_to_node(new_obj, val, key, None)
            else:
                raise KeyError(key)

        self.graph.add_node(name, model=new_obj)

        return new_obj


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
        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_neuron('2', 'HodgkinHuxley', states={'n':0., 'm':0., 'h':1.})
        >>> G.add_synapse('1->2', '1', '2', 'AlphaSynapse')

        Notes
        -----
        """

        attr = self._get_model_attributes(obj, id=id, **kwargs)

        self.graph.add_node(id, **attr)

        return self.graph.nodes[id]

    def _parse_id_and_attribute(self):
        pass

    def _set_single_input(self, id, attr, source):
        model = self.graph.nodes[id]['model']

        if type(source) == str:
            if source in self.graph.nodes:
                sid = source
                sattr = None
            else:
                seg = source.split('.')
                sid = sum(seg[:-1])
                sattr = seg[-1]
                if sid not in self.graph.nodes:
                    raise KeyError("Graph has no {} nor {}".format(source, id))
                if sattr not in model.vars:
                    raise KeyError("Model {} has no variable {}".format(
                        model.__name__, sattr))
        elif isinstance(source, Model):
            sid = source.id
            sattr = None
            if sid not in self.graph.nodes:
                raise KeyError("Graph has no {}".format(source))
        elif isinstance(source, _Variable):
            sid = source.id
            sattr = source.name
            if not sid in self.graph.nodes:
                raise KeyError("Graph has no {}".format(source))
            if sattr not in model.vars:
                raise KeyError("Model {} has no variable {}".format(
                    model.__name__, sattr))

        smodel = self.graph.nodes[sid]['model']

        if attr is None and sattr is None:
            _attr = list(sattr.inter.keys()) + list(sattr.state.keys())
            _attr = [x for x in model.input.keys() if x in _attr]

            if len(_attr) == 0:
                raise AttributeError("Fail to infer I/O pair;" + \
                    " {} and {}".format(model.__name__, smodel.__name__) + \
                    " have no common attribute.")

            elif len(_attr) > 1:
                raise AttributeError("Fail to infer I/O pair;" + \
                    " {} and {}".format(model.__name__, smodel.__name__) + \
                    " have more than one common attributes" + \
                    " {}".format(_attr))

            attr = sattr = _attr[0]

        elif attr is None and sattr is not None:
            if sattr not in model.input.keys():
                raise AttributeError("{} does not appear in".format(sattr) + \
                    " the input argument of {}".format(model.__name__))
            attr = sattr

        elif attr is not None and sattr is None:
            if attr not in smodel.state.keys() and \
                attr not in smodel.inter.keys():
                raise AttributeError("{} does not appear in".format(sattr) + \
                    " {}".format(model.__name__))

        self.graph.add_edge(id, sid, output=sattr, input=attr)

    def set_inputs(self, id, inputs):

        if type(inputs) is list or type(inputs) is tuple:
            for val in inputs:
                self._set_single_input(self, id, None, val)
        elif type(inputs) is dict:
            for key, val in inputs.items():
                self._set_single_input(self, id, key, val)
        else:
            self._set_single_input(self, id, None, inputs)

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
