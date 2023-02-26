Neurons and Layers
==================

Neurons in ``spikingnet`` are grouped in layers. Here we document
the basic layer interface as well as the types of layers implemented
in ``spikingnet``.


The layer interface
-------------------

All layers in ``spikingnet`` share the same underlying interface.
When a layer is added to a ``SpikingNet`` object through its
``add_layer`` method, it expects the layer to conform
to the following interface:

- Layers should have a ``__call__`` method implemented returning one
  output.
- Layers should have ``reset`` and ``update`` methods implemented. The
  ``reset`` method is called to return the network and the layer to
  its initial state. ``update`` methods are triggered during learning.


Neuron models
-------------

.. autoclass:: spikelearn.neurons.LIFLayer
   :special-members: __call__
   :members:
   :inherited-members:

.. autoclass:: spikelearn.neurons.SpikingLayer
   :special-members: __call__
   :members:
   :inherited-members:

.. autoclass:: spikelearn.neurons.SpikingRecLayer
   :special-members: __call__
   :members:
   :inherited-members:
