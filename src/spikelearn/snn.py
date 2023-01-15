"""Spiking neural network

This module implements a general spiking neural network comprising an
arbitrary collection of layers, inputs, and synapses. 


"""

from .streamnet import StreamNet, Element

class OldSpikingNet:
    """
    A class implementing arbitrary neural networks
    
    Imstances start with empty networks, which can be subsequently
    populated with layers, inputs, and synapses

    - Layers are objects taking a one or more inputs and returning an
      output. They should implement the  methods `__call__` and `reset`
      and an attribute `out` that preserves the results from the last
      computation.

    - Inputs define external inputs whose values are passed to the
      network through the `__call__` method.

    - Synapses connect one or more presynaptic layers with a postsynaptic
      layer. They should implement the methods `__call__`, `reset`, and
      `update`. The `update` method should be able to take two arguments,
      one representing the activity of the postsynaptic neuron and 
      an optional flag that triggers learning if it is an active synapse.

    A network can be constructed using a series of method that help define
    inputs, outputs, layers, and the synapses connecting them. The
    construction of the network is meant to be a one-off process. Deleting
    layers or synapses would most likely break the network or result
    in unpredictable behavior.

    """

    def __init__(self):
        """Initiates an empty network
        """

        self._layers = {}
        self.inputs = []
        self.outputs = []
        self.synapses = []
        self.pre_synapse = {}
        self.pos_synapse = {}
        self.layer_synapses = {}

    def add_layer(self, snl, name):
        """Adds a layer to the snn

        Args:
            snl : a layer of neurons
            name : the name of the layer
        
        Raises:
            ValueError: User tries to reuse an existing name
        """

        if name in self._layers.keys():
            raise ValueError("Layer {} already defined".format(name))

        self._layers[name] = snl
        self.layer_synapses[name] = []

    def add_synapse(self, pos_name, syn, *pre_names):
        """Adds a synapse between two or more network nodes
        
        A synapse has only one postsynaptic connection but can
        have more than one presynaptic connection to include the
        case of ternary synapses involving one or more modulatory
        inputs.

        Args:

            pos_name : name of the postsynaptic layer
            syn :  a synapse object
            pre_names : names of one or more presynaptic layers

        Raises:

            ValueError: pre or postsynaptic connection not identified.

        """

        if pos_name not in self._layers.keys():
            raise ValueError("Layer {} not defined.".format(pos_name))
        
        for name in pre_names:
            if name not in self.inputs and name not in self._layers.keys():
                raise ValueError("{} is not a layer or an input.".format(name))

        syn_ind = len(self.synapses)
        self.synapses.append(syn)
        self.pre_synapse[syn_ind] = list(pre_names)
        self.pos_synapse[syn_ind] = pos_name
        self.layer_synapses[pos_name].append(syn_ind)

    def add_input(self, name):
        """ Add a external input node to the network
        """
        self.inputs.append(name)

    def add_output(self, name):
        """Adds a node to the output list
        """
        self.outputs.append(name)

    def reset(self):
        """Resets all elements of the network
        
        Broadcasts a reset signal to all nodes and synapses
        in the network
        """
        for _ ,layer in self._layers.items():
            layer.reset()
        for syn in self.synapses:
            syn.reset()

    def __call__(self, *args, learn=True):
        """Advances the network a single timestep.

        Returns:
            A list with the declared network outputs

        """
        self.learn = learn
        if len(args) != len(self.inputs):
            raise ValueError("Wrong number of inputs in snn")
        i_dict = {name:a for name, a in zip(self.inputs, args)}
        out_synapse  = []
        for i, syn in enumerate(self.synapses):
            ul = []
            for c in self.pre_synapse[i]:
                if c in i_dict.keys():
                    ul.append(i_dict[c])
                else:
                    ul.append(self._layers[c].out)
            out_synapse.append(syn(*ul))

        for name, layer in self._layers.items():
            xl = [out_synapse[ind] for ind in self.layer_synapses[name]]
            layer(*xl)

        self._update()

        return [self._layers[name].out for name in self.outputs]


    def _update(self):
        for ind, syn in enumerate(self.synapses):
            syn.update(self._layers[self.pos_synapse[ind]].out, self.learn)


class NeuronElement(Element):

    def __init__(self, neuron):
        self._neuron = neuron
        self._synapses = []
        self._n_pre = []
        self.out = self._neuron.out
        self.total_syn_inputs = 0

    def add_synapse(self, synapse, n_pre=1):
        self._synapses.append(synapse)
        self._n_pre.append(n_pre)
        self.total_syn_inputs += n_pre

    def reset(self):
        self._neuron.reset()
        for syn in self._synapses:
            syn.reset()

    def __call__(self, *args):
        if len(self._synapses) == 0:
            self.out = self._neuron(*args)
        else:
            neuron_args = []
            n = 0
            for i, syn in enumerate(self._synapses):
                n_inputs = self._n_pre[i]
                if n_inputs == 1:
                    neuron_args.append(syn(args[n]))
                else:
                    neuron_args.append(syn(*args[n:(n+n_inputs)]))
                n += n_inputs
            self.out = self._neuron(*neuron_args)
        return self.out

    def update(self, learn):
        for syn in self._synapses:
            syn.update(self.out, learn)

   

class SpikingNet(StreamNet):
    """
    A class implementing arbitrary neural networks
    
    Imstances start with empty networks, which can be subsequently
    populated with layers, inputs, and synapses

    - Layers are objects taking a one or more inputs and returning an
      output. They should implement the  methods `__call__` and `reset`
      and an attribute `out` that preserves the results from the last
      computation.

    - Inputs define external inputs whose values are passed to the
      network through the `__call__` method.

    - Synapses connect one or more presynaptic layers with a postsynaptic
      layer. They should implement the methods `__call__`, `reset`, and
      `update`. The `update` method should be able to take two arguments,
      one representing the activity of the postsynaptic neuron and 
      an optional flag that triggers learning if it is an active synapse.

    A network can be constructed using a series of method that help define
    inputs, outputs, layers, and the synapses connecting them. The
    construction of the network is meant to be a one-off process. Deleting
    layers or synapses would most likely break the network or result
    in unpredictable behavior.

    """

    def __init__(self):
        """Initiates an empty network
        """
        self.is_synapse = {}
        self.pos_synapse = {}
        super().__init__()


    def add_layer(self, snl, name):
        """Adds a layer to the snn

        Args:
            snl : a layer of neurons
            name : the name of the layer
        
        """            

        self.add_element(name, NeuronElement(snl))


    def add_synapse(self, pos_name, syn, *pre_names):

        """Adds a synapse between two or more network nodes
        
        A synapse has only one postsynaptic connection but can
        have more than one presynaptic connection to include the
        case of ternary synapses involving one or more modulatory
        inputs.

        Args:

            pos_name : name of the postsynaptic layer
            syn :  a synapse object
            pre_names : names of one or more presynaptic layers

        Raises:

            ValueError: pre or postsynaptic connection not identified.

        """
     
        self._elements[pos_name].add_synapse(syn, len(pre_names))
        for i, name in enumerate(pre_names):
            self.add_el_input(pos_name, name, 1)


    def reset(self):
        """Resets all elements of the network
        
        Broadcasts a reset signal to all nodes and synapses
        in the network
        """
        self.broadcast("reset")

    def update(self, learn):
        self.broadcast("update", learn)

    def __call__(self, *args, learn=True):
        """Advances the network a single timestep.

        Returns:
            A list with the declared network outputs

        """
        self.learn = learn
        super().__call__(*args)
        self.update(learn)
        return self.out

    

class NewSpikingNet(StreamNet):
    """
    A class implementing arbitrary neural networks
    
    Imstances start with empty networks, which can be subsequently
    populated with layers, inputs, and synapses

    - Layers are objects taking a one or more inputs and returning an
      output. They should implement the  methods `__call__` and `reset`
      and an attribute `out` that preserves the results from the last
      computation.

    - Inputs define external inputs whose values are passed to the
      network through the `__call__` method.

    - Synapses connect one or more presynaptic layers with a postsynaptic
      layer. They should implement the methods `__call__`, `reset`, and
      `update`. The `update` method should be able to take two arguments,
      one representing the activity of the postsynaptic neuron and 
      an optional flag that triggers learning if it is an active synapse.

    A network can be constructed using a series of method that help define
    inputs, outputs, layers, and the synapses connecting them. The
    construction of the network is meant to be a one-off process. Deleting
    layers or synapses would most likely break the network or result
    in unpredictable behavior.

    """

    def __init__(self):
        """Initiates an empty network
        """
        self.is_synapse = {}
        self.pos_synapse = {}
        super().__init__()


    def add_layer(self, snl, name):
        """Adds a layer to the snn

        Args:
            snl : a layer of neurons
            name : the name of the layer
        
        """

        self.add_element(name, snl)
        self.is_synapse[name] = False

    def add_synapse(self, pos_name, syn, *pre_names):

        """Adds a synapse between two or more network nodes
        
        A synapse has only one postsynaptic connection but can
        have more than one presynaptic connection to include the
        case of ternary synapses involving one or more modulatory
        inputs.

        Args:

            pos_name : name of the postsynaptic layer
            syn :  a synapse object
            pre_names : names of one or more presynaptic layers

        Raises:

            ValueError: pre or postsynaptic connection not identified.

        """
     
        n = 1
        while True:
            syn_name = "pos_name_syn_{}".format(n)
            if syn_name in self._elements.keys():
                n += 1
            else:
                break

        self.add_element(syn_name, syn)
        self.is_synapse[syn_name] = True
        self.set_el_inputs(syn_name, *pre_names)
        self.add_el_input(pos_name, syn_name, 1)
        self.pos_synapse[syn_name] = pos_name


    def reset(self):
        """Resets all elements of the network
        
        Broadcasts a reset signal to all nodes and synapses
        in the network
        """
        self.broadcast_method("reset")


    def __call__(self, *args, learn=True):
        """Advances the network a single timestep.

        Returns:
            A list with the declared network outputs

        """
        self.learn = learn
        super().__call__(*args)
        self._update()
        return self.out


    def _update(self):
        for name, el in self._elements.items():
            if self.is_synapse[name]:
                el.update(self._elements[self.pos_synapse[name]].out, self.learn)
    


