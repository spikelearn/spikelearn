Neurons and Layers
==================

Background
----------

Neurons in SpikingNet are grouped in layers. All neurons share the same underlying interface, with Spikingnet
expecting neurons to be callable and have `reset` and `update` methods
defined.

Layers
------

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
