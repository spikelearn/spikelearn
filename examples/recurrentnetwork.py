"""
Explores a sparse recurrent network with excitatory and inhibitory neurons
"""

from spikelearn import SpikingNet, SpikingLayer, StaticSynapse
from spikelearn.generators import Poisson
import numpy as np

def create_sparse_static_synapse(n_in, n_out, weight, prob, syn_type):
    """Creates a sparse static synapse"""
    w_sp = np.random.random((n_out, n_in))
    w_sp = np.where(w_sp < prob, weight, 0)
    return StaticSynapse(n_in, n_out, w_sp, syn_type=syn_type)


def create_sparse_network(n_exc, n_inh, n_input, p0, pext, weights):
    """Creates a sparse spiking neural network"""

    exc_layer = SpikingLayer(n_exc, 4)
    inh_layer = SpikingLayer(n_inh, 4)

    sei = create_sparse_static_synapse(n_exc, n_inh, weights['W0ei'], p0, "exc")
    see = create_sparse_static_synapse(n_exc, n_exc, weights['W0ee'], p0, "exc")
    sie = create_sparse_static_synapse(n_inh, n_exc, weights['W0ie'], p0, "inh")
    s0e = create_sparse_static_synapse(n_input, n_exc, weights['W0ext'], pext, "exc")

    snn = SpikingNet()

    snn.add_input("input")
    snn.add_layer(exc_layer, "exc")
    snn.add_layer(inh_layer, "inh")
    snn.add_synapse("exc", s0e, "input")
    snn.add_synapse("exc", sie, "inh")
    snn.add_synapse("exc", see, "exc")
    snn.add_synapse("inh", sei, "exc")
    snn.add_output("exc")
    snn.add_output("inh")
    return snn


if __name__ == "__main__":

    import matplotlib.pyplot as pt

    d_weights = {
        "W0ee" : 0.1,
        "W0ei" : 0.2,
        "W0ie" : 0.5,
        "W0ext" : 0.3,
    }

    sp_snn = create_sparse_network(1000, 20, 200, 0.05, 0.05, d_weights)

    spikes = np.zeros(200)
    spike_gen = Poisson(100, 0.5)
    out_list = []
    out_inh = []

    for i in range(1000):
        spikes[:100] = spike_gen()
        oute, outi = sp_snn(spikes)
        out_list.append(np.mean(oute))
        out_inh.append(np.mean(outi))

    pt.plot(out_list)
    pt.show()
