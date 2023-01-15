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

class SpikingLayer(Layer):
    """Implements a leaky integrate and fire
    
    Implements a layer of LIF neurons. Its interface conforms
    to that of a `Layer` object. 

    Attributes:

        out: array containing last activations

        N: number of neurons in the layer

        v: array with membrane potential

        s: array with neuron activations

        tau: decay time, in timestep units

    """

    def __init__(self, N, tau, v0=1, refr=True):
        """Instantiates a layer of LIF neurons

        Args:

            N : number of neurons in the layer
            tau : decay time, in timestep units
            v0 (optional, default 1) : threshold value

        """
        self._N = N
        self._tau = tau
        self._a = np.exp(-1./tau)
        self._b = 1-self._a
        self._v0 = v0*np.ones(N)
        self._v = np.zeros(N)
        self._s = np.zeros(N)
        self._refr = refr

    @property
    def N(self):
        return self._N

    @property
    def tau(self):
        return self._tau

    @property
    def s(self):
        return self._s
    
    @property
    def v(self):
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
        """Advances the neuron a single timestep"""
        return self(x)

    def reset(self):
        """Resets the neuron internal state
        """
        self._v = np.zeros(self.N)
        self._s = np.zeros(self.N)

    @property
    def out(self):
        return self.s


class SpikingRecLayer(SpikingLayer):
    """Implements spiking neurons with an internal recurrent interaction.

    Implements a layer of leaky integrate and fire (LIF) neurons with
    an additional static, recurrent interaction within the layer. This
    layer can be used to implement cross-inhibition between neurons
    within the layer. Its interface conforms to that of a `Layer` object. 

    Attributes:

        out: array containing last activations

        N: number of neurons in the layer

        v: array with membrane potential

        s: array with neuron activations

        tau: decay time, in timestep units

        Wrec: recurrent synaptic weights

    """

    def __init__(self, N, tau, Wrec, v0=1):
        """Instantiates a layer of LIF neurons

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

