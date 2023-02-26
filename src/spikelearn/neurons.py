#Copyright Argonne 2022. See LICENSE.md for details.

"""
Neurons in SpikingNet are grouped in layers.

All neurons share the same underlying interface, with Spikingnet
expecting neurons to be callable and have `reset` and `update` methods
defined.

"""

from .layer import Layer

import numpy as np

def hard(x):
    """Implements a Heaviside or step function
    
    Args:
        x: a numpy array object

    Returns:
        An array: 1 if x>0, 0 otherwise

    """
    return np.where(x>=0, 1.0, 0.0).astype(x.dtype)

class LIFLayer(Layer):
    """Implements a leaky integrate and fire
    
    Implements a layer of LIF neurons. Its interface conforms
    to that of a `Layer` object. 

    Args:

        N : Neurons in the layer
        tau : decay time, in timestep units
        v0  : threshold value (optional, default 1)
        refr : boolean, neuron has 1 timestep refractory period

    """

    def __init__(self, N, tau, v0=1, refr=True):
        """Instantiates a layer of LIF neuron
        """

        self._N = N
        self._tau = tau
        self._a = np.exp(-1./tau)
        self._b = 1-self._a
        self._v0 = v0*np.ones(N)
        self._refr = refr
        self.reset()
        self._group_synapses = False

    @property
    def group_synapses(self):
        return self._group_synapses

    @property
    def N(self):
        """Neurons in the layer"""
        return self._N

    @property
    def tau(self):
        """Leakage time in timesteps"""
        return self._tau

    @property
    def s(self):
        """Output spikes"""
        return self._s
    
    @property
    def v(self):
        """Membrane potential"""
        return self._v

    def __call__(self, *x):
        """Advances the neuron a single timestep
        
        Args:
            x: a tuple of independent inputs

        Returns:
            An array of spikes, 1 if a neuron spikes 0 otherwise
        
        """

        xtot = sum(x)
        if self._refr:
            self._v = (1-self._s) * (self._a * self._v + self._b * xtot)
        else:
            self._v = (1-self._s) * self._a * self._v + self._b * xtot
        self._s = hard(self._v-self._v0)
        return self._s

    def step(self, *x):
        """Advances the neuron a single timestep
   
        Args:
            x: a tuple of independent inputs

        Returns:
            An array of spikes, 1 if a neuron spikes 0 otherwise
        """
 
        return self(x)

    def reset(self):
        """Resets the neuron internal state
        """
        self._v = np.zeros(self.N)
        self._s = np.zeros(self.N)

    @property
    def out(self):
        """Output spikes"""
        return self.s


SpikingLayer = LIFLayer

class SpikingRecLayer(LIFLayer):
    """Implements spiking neurons with an internal recurrent interaction.

    Implements a layer of leaky integrate and fire (LIF) neurons with
    an additional static, recurrent interaction within the layer. This
    layer can be used to implement cross-inhibition between neurons
    within the layer. Its interface conforms to that of a `Layer` object. 

    Args:

        N : number of neurons in the layer
        tau : decay time, in timestep units
        Wrec: a 2D array with synaptic weights
        v0 (optional, default 1) : threshold value

    """

    def __init__(self, N, tau, Wrec, v0=1):
        """Instantiates a layer of LIF neurons with recurrent weights

        Args:

            N : number of neurons in the layer
            tau : decay time, in timestep units
            Wrec: a 2D array with synaptic weights
            v0 (optional, default 1) : threshold value

        """

        self.Wrec = Wrec
        super().__init__(N, tau, v0)


    def __call__(self, *x):
        """Advances the neuron a single timestep
        
        Args:
            x: a tuple of independent inputs

        Returns:
            An array of spikes, 1 if a neuron spikes 0 otherwise
        
        """
        xn = x + (self.Wrec @ self.s,)
        return super().__call__(*xn)



class BioLIFLayer(Layer):

    def __init__(self, N, nudt, v0=0.5):
        self.N = N
        self.nu0 = nudt
        self.v0 = v0
        self.reset()
        self._group_synapses = True

    @property
    def group_synapses(self):
        return self._group_synapses

    def reset(self):
        self.v = np.zeros(self.N)
        self.vold = np.zeros(self.N)
        self.s = np.zeros(self.N)


    def __call__(self, xe, xi, perf=False):
        self.nu = (1+xe+xi)
        a = np.exp(-self.nu0*self.nu)
        self.de = xe/self.nu
        self.v = self.vold * a + self.de*(1-a)
        self.s = hard(self.v-self.v0)
        if perf:
            self.xe = xe
            self.calc_perf()
        self.vold = (1-self.s)*self.v
        return self.s

    @property
    def out(self):
        return self.s

    def calc_perf(self):
        dv = self.v - self.vold
        av = 0.5*(self.v + self.vold)
        self.power = self.xe*((1-self.de)**2 + dv/self.nu0*(2-self.de-av))

