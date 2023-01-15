from spikelearn import SpikingNet, SpikingLayer, STDPSynapse
from spikelearn.generators import Poisson
import numpy as np

snn = SpikingNet()
sl = SpikingLayer(10, 4)
W0 = np.random.random((10,20))

syn = STDPSynapse(20, 10, np.copy(W0),
    tre=(0.5, 0.5), tro=(0.5, 0.5),
    rule_params={"Ap":0.1, "An":0.1},
    syn_type="exc")

snn.add_input("input1")
snn.add_layer(sl, "l1")
snn.add_synapse("l1", syn, "input1")
snn.add_output("l1")

u = Poisson(20, np.random.random(20))
for i in range(200):
    s = snn(u())
print(syn.W-W0)

