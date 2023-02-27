#Copyright Argonne 2022. See LICENSE.md for details.

from streamnet import StreamNet, Node
 

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


    def add_node(self, name, node):
        """Adds a layer to the snn

        Args:
            name : the name of the layer
            node : a neuron node
        
        """            
        super().add_node(name,
            Node(node.total_syn_inputs, 1, node, node.out))


    def add_pre_connections(self, name, name_list):
        con_list = []
        for pre_name in name_list:
            if self.node_exists(pre_name):
                con_list.append([pre_name, 1])
            elif self.input_exists(pre_name):
                con_list.append([pre_name])
            else:
                raise ValueError("{} not a valid node or input".format(pre_name))
        
        self.set_node_inputs(name, con_list)

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
