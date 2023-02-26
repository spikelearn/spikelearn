
Motivation
----------

We needed a SNN model with the following requirements:

- Capable of handling traditional ML workflows
- Heterogeneous, with the ability to integrate both mathematical models and
  neurons or synapses inspired on neuromorphic computing and emergent devices
- That could be easily parametrizable, in order to explore a large number of
  configurations in high performance computing environments.
- That could reproduce models in existing neuromorphic chips such as Loihi.
- That could handle neuromodulators and other neuroscience-inspired goodies.
- That could be easily extensible.
- That is capable of online learning through a variety of synaptic plasticity
  rules.


Spikelearn intends to fill that role.


Status
------

Spikelearn is still in development.

Quick install
-------------

The easiest way is directly through ``pypi``:

.. code::

    pip install spikelearn

Alternatively, you can directly obtain ``spikelearn`` through its 
`github repository <https://github.com/spikelearn/spikelearn>`_.


Usage
-----

.. code::

    from spikelearn import SpikingNet, SpikingLayer, StaticSynapse
    import numpy as np

    snn = SpikingNet()
    sl = SpikingLayer(10, 4)
    syn = StaticSynapse(10, 10, np.random.random((10,10)))

    snn.add_input("input1")
    snn.add_layer(sl, "l1")
    snn.add_synapse("l1", syn, "input1")
    snn.add_output("l1")

    u = 2*np.random.random(10)
    for i in range(10):
        s = snn(2*np.random.random(10))
        print(s)


Citing
------

If you want to acknowledge ``spikelearn`` in a publication, you can cite
the following work:

Angel Yanguas-Gil and Sandeep Madireddy, *AutoML for neuromorphic computing
and application-driven co-design: asynchronous, massively parallel
optimization of spiking architectures*, International Conference
on Rebooting Computing, San Francisco, CA (2022).

Acknowledgements
----------------

* Threadwork, U.S. Department of Energy Office of Science, 
  Microelectronics Program.


Copyright and license
---------------------

Copyright © 2022, UChicago Argonne, LLC

Spikelearn is distributed under the terms of BSD License. See `License file
<https://github.com/spikelearn/spikelearn/blob/master/LICENSE.md>`_.

Argonne Patent & Intellectual Property File Number: SF-22-154


