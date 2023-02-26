#Copyright Argonne 2022. See LICENSE.md for details.

from .streamnet import StreamNet, Element


class NeuronElement(Element):

    def __init__(self, neuron):
        self._neuron = neuron
        self._synapses = []
        self._syn_types = []
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
            if self._neuron.group_synapses:
                xe_list = []
                xi_list = []
                n = 0
                for i, syn in enumerate(self._synapses):
                    n_inputs = self._n_pre[i]
                    if syn.syn_type == 'exc':
                        ap_list = xe_list
                    elif syn.syn_type == 'inh':
                        ap_list = xi_list
                    else:
                        raise ValueError("Synapse type must be either exc or inh")
                    if n_inputs == 1:
                        ap_list.append(syn(args[n]))
                    else:
                        ap_list.append(syn(*args[n:(n+n_inputs)]))
                    n += n_inputs
                if len(xe_list) > 0:
                    xe = sum(xe_list)
                else:
                    xe = None
                if len(xi_list) > 0:
                    xi = sum(xi_list)
                else:
                    xi = None
                self.out = self._neuron(xe, xi)

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
    
    Instances start with empty networks, which can be subsequently
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
        """Broadcasts a reset signal to all layers and synapses
        in the network
        """
        self.broadcast("reset")

    def update(self, learn):
        """Broadcasts a learn signal to all layers and synapses"""
        self.broadcast("update", learn)

    def __call__(self, *args, learn=True):
        """Advances the network a single timestep.

        Args:
            args: a tuple of inputs. Must match the number of inputs in the
                SpikingNet object
            learn: if True, broadcasts a learn signal at the end of the
                timestep

        Returns:
            A list with the declared network outputs

        """
        
        self.learn = learn
        super().__call__(*args)
        self.update(learn)
        return self.out
