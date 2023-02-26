from spikelearn import SpikingNet, SpikingLayer, MSESynapse
from spikelearn.generators import Poisson
import numpy as np

snn = SpikingNet()
sl = SpikingLayer(10, 4)
W0 = np.random.random((10,20))
syn = MSESynapse(20, 10, W0,
    tre=(0.5, 0.5), tro=(0.5, 0.5), trm=(0.5,0.5),
    rule_params={"lr":0.01},
    syn_type="exc")

snn.add_input("input1")
snn.add_input("mod1")
snn.add_layer(sl, "l1")
snn.add_synapse("l1", syn, "input1", "mod1")
snn.add_output("l1")

u = Poisson(20, np.random.random(20))
for i in range(200):
    s = snn(u(), np.ones(10))
#    print(s)
print(syn.W)

