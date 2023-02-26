from spikelearn import SpikingNet, SpikingLayer, StaticSynapse
import numpy as np

np.random.seed(0)

snn = SpikingNet()
snn.add_input("input1")


sl = SpikingLayer(10, 4)
snn.add_layer(sl, "l1")

syn = StaticSynapse(10, 10, np.eye(10))
snn.add_synapse("l1", syn, "input1")

snn.add_output("l1")

analog_input = np.arange(0,2,0.2)
for i in range(10):
    s = snn(analog_input)
    print(s)

