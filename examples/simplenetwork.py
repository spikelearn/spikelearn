from spikelearn import SpikingNet, SpikingLayer, StaticSynapse
import numpy as np

np.random.seed(0)

snn = SpikingNet()
sl = SpikingLayer(10, 4)
syn = StaticSynapse(10, 10, np.eye(10))

snn.add_input("input1")
snn.add_layer(sl, "l1")
snn.add_synapse("l1", syn, "input1")
snn.add_output("l1")

a = np.arange(0,2,0.2)
print(a)
for i in range(10):
    s = snn(a)
    print(s)

